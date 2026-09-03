from __future__ import annotations

from vtkweb.catalog import AlgorithmCatalog
from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering import RenderManager


def initialize_app_controller(
    server,
    pipeline: PipelineGraph,
    rendering: RenderManager,
    catalog: AlgorithmCatalog,
) -> None:
    state = server.state
    ctrl = server.controller

    # -------------------------------------------------------------------------
    # Active node
    # -------------------------------------------------------------------------

    active_node = (
        pipeline.active_node
    )

    state.active_node_id = (
        active_node.id
        if active_node is not None
        else None
    )

    state.active_node_name = (
        active_node.name
        if active_node is not None
        else ""
    )

    state.active_node_type = (
        active_node.algorithm.GetClassName()
        if active_node is not None
        else ""
    )

    # -------------------------------------------------------------------------
    # Output-port visibility
    # -------------------------------------------------------------------------

    def update_output_visibility_state() -> None:
        view_id = (
            rendering.active_view_id
        )

        result = {}

        for node in pipeline.nodes.values():
            output_count = (
                node.algorithm
                .GetNumberOfOutputPorts()
            )

            result[node.id] = {
                str(port): (
                    rendering.output_visible_in_view(
                        node.id,
                        port,
                        view_id,
                    )
                )
                for port in range(
                    output_count
                )
            }

        state.output_port_visibility = (
            result
        )

    update_output_visibility_state()

    # Keep this alias temporarily so any existing code
    # which asks to refresh pipeline visibility does not
    # immediately break.
    ctrl.update_node_visibility_state = (
        update_output_visibility_state
    )

    # -------------------------------------------------------------------------
    # Refresh
    # -------------------------------------------------------------------------

    def refresh_node(
        node_id: str,
    ) -> None:
        ctrl.update_properties_state(
            node_id
        )

        ctrl.update_representation_state(
            node_id
        )

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

        node = (
            pipeline.active_node
        )

        with state:
            state.active_node_id = (
                node.id
            )

            state.active_node_name = (
                node.name
            )

            state.active_node_type = (
                node.algorithm.GetClassName()
            )

        refresh_node(
            node.id
        )

    # -------------------------------------------------------------------------
    # Output-port interaction
    # -------------------------------------------------------------------------

    def output_port_click(
        node_id: str,
        output_port: int,
        shift_key: bool = False,
    ) -> None:
        output_port = int(
            output_port
        )

        # A port click always makes its node active.
        set_active_node(
            node_id
        )

        if not shift_key:
            return

        rendering.toggle_output_in_view(
            node_id,
            output_port,
            rendering.active_view_id,
        )

        update_output_visibility_state()

        ctrl.update_representation_state(
            node_id
        )

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Create source / filter
    # -------------------------------------------------------------------------

    def create_filter(
        class_name: str,
    ) -> None:
        descriptor = next(
            item
            for item in catalog.algorithms
            if item.class_name
            == class_name
        )

        algorithm = catalog.create(
            class_name
        )

        previous_active = (
            pipeline.active_node
        )

        node = pipeline.add_node(
            algorithm,
            name=descriptor.label,
            visible=True,
        )

        # Default representation remains output 0.
        rendering.add_representation(
            node.id,
            output_port=0,
            kind="surface",
            view_ids={
                rendering.active_view_id
            },
        )

        ctrl.pipeline_add_node(
            node
        )

        if (
            algorithm.GetNumberOfInputPorts()
            > 0
            and previous_active is not None
        ):
            edge = pipeline.connect(
                previous_active.id,
                node.id,
                source_port=0,
                target_port=0,
            )

            ctrl.pipeline_add_edge(
                edge
            )

        update_output_visibility_state()

        ctrl.close_filter_browser()

        algorithm.Update()

        set_active_node(
            node.id
        )

        rendering.reset_camera()

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Delete node
    # -------------------------------------------------------------------------

    def delete_active_node() -> None:
        node = (
            pipeline.active_node
        )

        if node is None:
            return

        node_id = (
            node.id
        )

        incident_edges = [
            edge
            for edge in pipeline.edges
            if (
                edge.source_node_id
                == node_id
                or edge.target_node_id
                == node_id
            )
        ]

        for edge in incident_edges:
            ctrl.pipeline_remove_edge(
                edge
            )

        rendering.remove_node(
            node_id
        )

        pipeline.remove_node(
            node_id
        )

        ctrl.pipeline_remove_node(
            node_id
        )

        update_output_visibility_state()

        if (
            pipeline.active_node
            is not None
        ):
            set_active_node(
                pipeline.active_node.id
            )

        else:
            with state:
                state.active_node_id = None
                state.active_node_name = ""
                state.active_node_type = ""

            ctrl.update_properties_state(
                None
            )

            ctrl.update_representation_state(
                None
            )

            ctrl.view_update()

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    ctrl.refresh_node = (
        refresh_node
    )

    ctrl.set_active_node = (
        set_active_node
    )

    ctrl.output_port_click = (
        output_port_click
    )

    ctrl.update_output_visibility_state = (
        update_output_visibility_state
    )

    ctrl.create_filter = (
        create_filter
    )

    ctrl.delete_active_node = (
        delete_active_node
    )

    server.trigger(
        "delete_active_node"
    )(
        delete_active_node
    )
