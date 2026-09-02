from __future__ import annotations

import vtk

from trame.ui.vuetify3 import (
    SinglePageWithDrawerLayout,
)
from trame.widgets import client
from trame.widgets import vuetify3 as v3

from vtkweb.catalog import AlgorithmCatalog
from vtkweb.input_arrays import (
    inspect_input_arrays,
    set_input_array,
)
from vtkweb.pipeline import PipelineGraph
from vtkweb.properties import (
    inspect_properties,
    set_property,
)
from vtkweb.rendering import RenderManager
from vtkweb.ui.filter_browser import (
    FILTER_BROWSER_STYLE,
    build_filter_browser,
    initialize_filter_browser,
)
from vtkweb.ui.pipeline_view import (
    build_pipeline_view,
)
from vtkweb.ui.properties_view import (
    PROPERTY_STYLE,
    build_properties_view,
)
from vtkweb.ui.render_view import (
    build_render_view,
)


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
    state.input_arrays = []

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

    initialize_filter_browser(
        state,
        catalog,
    )

    # -------------------------------------------------------------------------
    # Synchronization
    # -------------------------------------------------------------------------

    def update_property_state(
        node_id: str,
    ) -> None:
        algorithm = (
            pipeline.nodes[node_id].algorithm
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
                for prop in inspect_properties(
                    algorithm
                )
            ]

            state.input_arrays = [
                {
                    "index": item.index,
                    "label": item.label,
                    "value": item.value,
                    "items": item.items,
                }
                for item in inspect_input_arrays(
                    algorithm
                )
            ]

    def update_representation_state(
        node_id: str,
    ) -> None:
        arrays = rendering.get_arrays(
            node_id
        )

        representation = (
            rendering.representations[node_id]
        )

        items = [
            {
                "title": f"{name} (Point)",
                "value": f"point:{name}",
            }
            for name in arrays["point"]
        ]

        items += [
            {
                "title": f"{name} (Cell)",
                "value": f"cell:{name}",
            }
            for name in arrays["cell"]
        ]

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
                (
                    state.color_range_min,
                    state.color_range_max,
                ) = representation.scalar_range

    def refresh_node(
        node_id: str,
    ) -> None:
        update_property_state(node_id)
        update_representation_state(node_id)
        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Active node
    # -------------------------------------------------------------------------

    def set_active_node(
        node_id: str,
    ) -> None:
        pipeline.set_active_node(
            node_id
        )

        node = pipeline.active_node

        with state:
            state.active_node_id = node.id
            state.active_node_name = node.name
            state.active_node_type = (
                node.algorithm.GetClassName()
            )

        refresh_node(node.id)

    # -------------------------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------------------------

    def toggle_visibility(
        node_id: str,
    ) -> None:
        rendering.toggle_visibility(
            node_id
        )

        state.node_visibility = {
            **state.node_visibility,
            node_id: pipeline.nodes[
                node_id
            ].visible,
        }

        ctrl.view_update()

    def node_click(
        node_id: str,
        shift_key: bool = False,
    ) -> None:
        if shift_key:
            toggle_visibility(
                node_id
            )
        else:
            set_active_node(
                node_id
            )

    def delete_active_node() -> None:
        node = pipeline.active_node

        if node is None:
            return

        node_id = node.id

        for edge in list(pipeline.edges):
            if (
                edge.source_node_id == node_id
                or edge.target_node_id == node_id
            ):
                ctrl.pipeline_remove_edge(
                    edge
                )

        rendering.remove_representation(
            node_id
        )

        pipeline.remove_node(
            node_id
        )

        ctrl.pipeline_remove_node(
            node_id
        )

        visibility = dict(
            state.node_visibility
        )

        visibility.pop(
            node_id,
            None,
        )

        state.node_visibility = visibility

        if pipeline.active_node is not None:
            set_active_node(
                pipeline.active_node.id
            )
        else:
            with state:
                state.active_node_id = None
                state.active_node_name = ""
                state.active_node_type = ""
                state.filter_properties = []
                state.input_arrays = []
                state.color_array_items = []
                state.color_array = None

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Filter browser
    # -------------------------------------------------------------------------

    def open_filter_browser() -> None:
        with state:
            state.filter_browser_query = ""
            state.filter_browser_items = (
                state.filter_catalog_items
            )
            state.filter_browser_selected = 0
            state.filter_browser_open = True

    def close_filter_browser() -> None:
        state.filter_browser_open = False

    def set_filter_browser_query(
        query: str,
    ) -> None:
        query = query or ""
        needle = query.casefold()

        items = [
            item
            for item in state.filter_catalog_items
            if (
                not needle
                or needle in item["title"].casefold()
                or needle
                in item["class_name"].casefold()
            )
        ]

        with state:
            state.filter_browser_query = query
            state.filter_browser_items = items
            state.filter_browser_selected = 0

    def set_filter_browser_selected(
        index: int,
    ) -> None:
        state.filter_browser_selected = int(
            index
        )

    def filter_browser_keydown(
        key: str,
    ) -> None:
        items = state.filter_browser_items

        if key == "ArrowDown" and items:
            state.filter_browser_selected = min(
                state.filter_browser_selected + 1,
                len(items) - 1,
            )

        elif key == "ArrowUp" and items:
            state.filter_browser_selected = max(
                state.filter_browser_selected - 1,
                0,
            )

        elif key == "Enter" and items:
            create_filter(
                items[
                    state.filter_browser_selected
                ]["value"]
            )

        elif key == "Escape":
            close_filter_browser()

    # -------------------------------------------------------------------------
    # Source / filter creation
    # -------------------------------------------------------------------------

    def create_filter(
        class_name: str,
    ) -> None:
        descriptor = next(
            item
            for item in catalog.algorithms
            if item.class_name == class_name
        )

        algorithm = catalog.create(
            class_name
        )

        previous_active = (
            pipeline.active_node
        )

        new_node = pipeline.add_node(
            algorithm,
            name=descriptor.label,
            visible=True,
        )

        rendering.add_representation(
            new_node.id
        )

        ctrl.pipeline_add_node(
            new_node
        )

        if algorithm.GetNumberOfInputPorts():
            edge = pipeline.connect(
                previous_active.id,
                new_node.id,
            )

            ctrl.pipeline_add_edge(
                edge
            )

        state.node_visibility = {
            **state.node_visibility,
            new_node.id: new_node.visible,
        }

        state.filter_browser_open = False

        pipeline.set_active_node(
            new_node.id
        )

        algorithm.Update()

        set_active_node(
            new_node.id
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

        refresh_node(
            pipeline.active_node_id
        )

    # -------------------------------------------------------------------------
    # Generic properties
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

        refresh_node(
            pipeline.active_node_id
        )

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
        descriptor = get_descriptor(name)

        values = list(
            descriptor.value
        )

        values[int(index)] = float(value)

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

        descriptor = get_descriptor(name)

        values = list(
            descriptor.value
        )

        values[int(index)] = float(value)

        apply_property(
            descriptor,
            values,
        )

    def add_filter_list_value(
        name: str,
    ) -> None:
        descriptor = get_descriptor(name)

        values = list(
            descriptor.value
        )

        values.append(
            values[-1] if values else 0.0
        )

        apply_property(
            descriptor,
            values,
        )

    def remove_filter_list_value(
        name: str,
        index: int,
    ) -> None:
        descriptor = get_descriptor(name)

        values = list(
            descriptor.value
        )

        del values[int(index)]

        apply_property(
            descriptor,
            values,
        )

    # -------------------------------------------------------------------------
    # Elevation
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

        refresh_node(
            pipeline.active_node_id
        )

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def set_representation_mode(
        value: str,
    ) -> None:
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
        node_id = (
            pipeline.active_node_id
        )

        if not value:
            rendering.set_array(
                node_id,
                None,
            )

            state.color_array = None
            ctrl.view_update()
            return

        association, array_name = (
            value.split(":", 1)
        )

        rendering.set_array(
            node_id,
            array_name,
            association,
        )

        update_representation_state(
            node_id
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
        node_id = (
            pipeline.active_node_id
        )

        representation = (
            rendering.representations[
                node_id
            ]
        )

        if representation.array_name is None:
            return

        scalar_range = (
            rendering.get_array_range(
                node_id,
                representation.array_name,
                representation.association,
            )
        )

        if scalar_range is None:
            return

        rendering.set_scalar_range(
            node_id,
            *scalar_range,
        )

        with state:
            (
                state.color_range_min,
                state.color_range_max,
            ) = scalar_range

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    ctrl.node_click = node_click
    ctrl.toggle_visibility = toggle_visibility
    ctrl.delete_active_node = delete_active_node

    ctrl.open_filter_browser = open_filter_browser
    ctrl.close_filter_browser = close_filter_browser
    ctrl.create_filter = create_filter

    ctrl.set_filter_browser_query = (
        set_filter_browser_query
    )
    ctrl.set_filter_browser_selected = (
        set_filter_browser_selected
    )
    ctrl.filter_browser_keydown = (
        filter_browser_keydown
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

    server.trigger(
        "open_filter_browser"
    )(
        open_filter_browser
    )

    server.trigger(
        "delete_active_node"
    )(
        delete_active_node
    )

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
                    title="Add Source / Filter",
                    prepend_icon="mdi-plus",
                    click=ctrl.open_filter_browser,
                )

        with layout.content:
            client.Style(
                PROPERTY_STYLE
            )

            client.Style(
                FILTER_BROWSER_STYLE
            )

            client.ClientTriggers(
                mounted="""
                    window.__vtkwebGlobalKeydown = (event) => {
                        const target = event.target;
                        const tag =
                            target?.tagName?.toLowerCase();

                        const editing =
                            tag === 'input' ||
                            tag === 'textarea' ||
                            tag === 'select' ||
                            target?.isContentEditable;

                        if (
                            event.ctrlKey &&
                            event.code === 'Space'
                        ) {
                            event.preventDefault();
                            trigger('open_filter_browser');
                            return;
                        }

                        if (
                            event.key === 'Delete' &&
                            !editing &&
                            !filter_browser_open
                        ) {
                            event.preventDefault();
                            trigger('delete_active_node');
                        }
                    };

                    window.addEventListener(
                        'keydown',
                        window.__vtkwebGlobalKeydown
                    );
                """,
                before_unmount="""
                    if (window.__vtkwebGlobalKeydown) {
                        window.removeEventListener(
                            'keydown',
                            window.__vtkwebGlobalKeydown
                        );

                        delete window.__vtkwebGlobalKeydown;
                    }
                """,
            )

            build_filter_browser(
                state,
                ctrl,
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
