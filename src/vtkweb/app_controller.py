from __future__ import annotations

from collections.abc import Iterable

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
    # Primitive application commands
    # -------------------------------------------------------------------------

    def create_node(
        class_name: str,
        *,
        name: str | None = None,
        node_id: str | None = None,
    ) -> str:
        descriptor = next(
            item for item in catalog.algorithms if item.class_name == class_name
        )

        processor = catalog.create(class_name)

        node = pipeline.add_node(
            processor,
            name=name or descriptor.label,
            node_id=node_id,
        )

        ctrl.pipeline_add_node(node)

        return node.id

    def connect_nodes(
        source_node_id: str,
        target_node_id: str,
        *,
        source_port: int = 0,
        target_port: int = 0,
    ) -> None:
        edge = pipeline.connect(
            source_node_id,
            target_node_id,
            source_port=int(source_port),
            target_port=int(target_port),
        )

        ctrl.pipeline_add_edge(edge)

    def set_node_property(
        node_id: str,
        name: str,
        value,
    ) -> None:
        pipeline.set_property(
            node_id,
            name,
            value,
        )
        ctrl.update_representation_state(node_id)
        ctrl.view_update()

    def set_node_vector_component(
        node_id: str,
        name: str,
        index: int,
        value,
    ) -> None:
        pipeline.set_vector_component(
            node_id,
            name,
            int(index),
            value,
        )
        ctrl.update_representation_state(node_id)
        ctrl.view_update()

    def set_node_list_value(
        node_id: str,
        name: str,
        index: int,
        value,
    ) -> None:
        pipeline.set_list_value(
            node_id,
            name,
            int(index),
            value,
        )
        ctrl.update_representation_state(node_id)
        ctrl.view_update()

    def add_node_list_value(
        node_id: str,
        name: str,
    ) -> None:
        pipeline.add_list_value(
            node_id,
            name,
        )
        ctrl.update_representation_state(node_id)
        ctrl.view_update()

    def remove_node_list_value(
        node_id: str,
        name: str,
        index: int,
    ) -> None:
        pipeline.remove_list_value(
            node_id,
            name,
            int(index),
        )
        ctrl.update_representation_state(node_id)
        ctrl.view_update()

    def set_node_input_array(
        node_id: str,
        index: int,
        value,
    ) -> None:
        pipeline.set_input_array(
            node_id,
            int(index),
            value,
        )
        ctrl.update_representation_state(node_id)
        ctrl.view_update()

    def add_representation(
        node_id: str,
        output_port: int = 0,
        kind: str = "surface",
        view_ids: Iterable[str] = (),
        representation_id: str | None = None,
    ) -> str:
        representation = rendering.add_representation(
            node_id,
            output_port=int(output_port),
            kind=kind,
            view_ids=view_ids,
            representation_id=representation_id,
        )
        ctrl.update_representation_state(node_id)
        ctrl.view_update()
        return representation.id

    # -------------------------------------------------------------------------
    # Refresh / synchronization
    # -------------------------------------------------------------------------

    def refresh_node(
        node_id: str,
    ) -> None:
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
            state.active_view_id,
        )

        ctrl.update_representation_state(node_id)

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Interactive node insertion
    # -------------------------------------------------------------------------

    def insert_node(
        class_name: str,
    ) -> None:
        previous_active = pipeline.active_node

        node_id = create_node(class_name)
        node = pipeline.nodes[node_id]
        processor = node.processor

        if processor.GetNumberOfInputPorts() > 0 and previous_active is not None:
            connect_nodes(
                previous_active.id,
                node_id,
                source_port=0,
                target_port=0,
            )

        if processor.GetNumberOfOutputPorts() > 0:
            add_representation(
                node_id,
                output_port=0,
                kind="outline",
                view_ids={state.active_view_id},
            )

        ctrl.close_node_browser()

        # At this point all required automatic input connections have been
        # established. Node creation itself intentionally never inspects or
        # executes a processor with mandatory inputs.
        processor.Update()

        pipeline.sync_node_from_runtime(node_id)

        set_active_node(node_id)

        rendering.reset_camera()

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Delete node
    # -------------------------------------------------------------------------

    def delete_node(
        node_id: str,
    ) -> None:
        if node_id not in pipeline.nodes:
            return

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

    def delete_active_node() -> None:
        node_id = pipeline.active_node_id
        if node_id is not None:
            delete_node(node_id)

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    ctrl.create_node = create_node
    ctrl.connect_nodes = connect_nodes
    ctrl.set_node_property = set_node_property
    ctrl.set_node_vector_component = set_node_vector_component
    ctrl.set_node_list_value = set_node_list_value
    ctrl.add_node_list_value = add_node_list_value
    ctrl.remove_node_list_value = remove_node_list_value
    ctrl.set_node_input_array = set_node_input_array
    ctrl.add_representation = add_representation
    ctrl.refresh_node = refresh_node
    ctrl.sync_node_from_runtime = sync_node_from_runtime
    ctrl.set_active_node = set_active_node
    ctrl.output_port_click = output_port_click
    ctrl.insert_node = insert_node
    ctrl.delete_node = delete_node
    ctrl.delete_active_node = delete_active_node

    server.trigger("delete_active_node")(delete_active_node)
