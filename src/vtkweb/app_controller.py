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
    # Refresh / synchronization
    # -------------------------------------------------------------------------

    def refresh_node(
        node_id: str,
    ) -> None:
        # Core node/representation/view data already lives in trame state. The
        # only inspector refresh still needed here is output-array metadata,
        # which depends on executing/inspecting VTK output data.
        ctrl.update_representation_state(node_id)
        ctrl.view_update()

    def sync_node_from_runtime(
        node_id: str,
    ) -> None:
        pipeline.sync_node_from_runtime(node_id)
        refresh_node(node_id)

    # -------------------------------------------------------------------------
    # Active node
    # -------------------------------------------------------------------------

    def set_active_node(
        node_id: str,
    ) -> None:
        pipeline.set_active_node(node_id)

        # Output-port selection is inspector-local state. Reset it when the
        # active algorithm changes so it can never point past the new node's
        # available outputs.
        state.active_representation_output_port = 0
        ctrl.update_representation_state(node_id)

    # -------------------------------------------------------------------------
    # Output-port interaction
    # -------------------------------------------------------------------------

    def output_port_click(
        node_id: str,
        output_port: int,
        shift_key: bool = False,
    ) -> None:
        output_port = int(output_port)

        set_active_node(node_id)

        if not shift_key:
            return

        rendering.toggle_output_in_view(
            node_id,
            output_port,
            rendering.active_view_id,
        )

        ctrl.update_representation_state(node_id)
        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Create source / filter
    # -------------------------------------------------------------------------

    def create_filter(
        class_name: str,
    ) -> None:
        descriptor = next(
            item for item in catalog.algorithms if item.class_name == class_name
        )

        algorithm = catalog.create(class_name)

        previous_active = pipeline.active_node

        node = pipeline.add_node(
            algorithm,
            name=descriptor.label,
        )

        ctrl.pipeline_add_node(node)

        if algorithm.GetNumberOfInputPorts() > 0 and previous_active is not None:
            edge = pipeline.connect(
                previous_active.id,
                node.id,
                source_port=0,
                target_port=0,
            )
            ctrl.pipeline_add_edge(edge)

        if algorithm.GetNumberOfOutputPorts() > 0:
            rendering.add_representation(
                node.id,
                output_port=0,
                kind="surface",
                view_ids={rendering.active_view_id},
            )

        ctrl.close_filter_browser()

        algorithm.Update()
        pipeline.sync_node_from_runtime(node.id)

        set_active_node(node.id)

        rendering.reset_camera()
        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Delete node
    # -------------------------------------------------------------------------

    def delete_active_node() -> None:
        node = pipeline.active_node

        if node is None:
            return

        node_id = node.id

        incident_edges = [
            edge
            for edge in pipeline.edges
            if (edge.source_node_id == node_id or edge.target_node_id == node_id)
        ]

        for edge in incident_edges:
            ctrl.pipeline_remove_edge(edge)

        rendering.remove_node(node_id)
        pipeline.remove_node(node_id)
        ctrl.pipeline_remove_node(node_id)

        if pipeline.active_node is not None:
            set_active_node(pipeline.active_node.id)
        else:
            state.active_representation_output_port = 0
            ctrl.update_representation_state(None)

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    ctrl.refresh_node = refresh_node
    ctrl.sync_node_from_runtime = sync_node_from_runtime
    ctrl.set_active_node = set_active_node
    ctrl.output_port_click = output_port_click
    ctrl.create_filter = create_filter
    ctrl.delete_active_node = delete_active_node

    server.trigger("delete_active_node")(delete_active_node)
