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
    # Pipeline-view visibility state
    # -------------------------------------------------------------------------

    def update_node_visibility_state() -> None:
        view_id = (
            rendering.active_view_id
        )

        state.node_visibility = {
            node.id: (
                rendering.node_visible_in_view(
                    node.id,
                    view_id,
                )
            )
            for node in pipeline.nodes.values()
        }

    update_node_visibility_state()

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
    # Node visibility in active view
    # -------------------------------------------------------------------------

    def toggle_visibility(
        node_id: str,
    ) -> None:
        rendering.toggle_node_in_view(
            node_id,
            rendering.active_view_id,
        )

        update_node_visibility_state()

        if (
            node_id
            == pipeline.active_node_id
        ):
            ctrl.update_representation_state(
                node_id
            )

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

    # -------------------------------------------------------------------------
    # Create source / filter
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

        node = pipeline.add_node(
            algorithm,
            name=descriptor.label,
            visible=True,
        )

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

        update_node_visibility_state()

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

        node_id = node.id

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

        update_node_visibility_state()

        if pipeline.active_node is not None:
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

    ctrl.update_node_visibility_state = (
        update_node_visibility_state
    )

    ctrl.toggle_visibility = (
        toggle_visibility
    )

    ctrl.node_click = (
        node_click
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
