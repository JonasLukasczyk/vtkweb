from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from vtkweb.catalog import AlgorithmCatalog
from vtkweb.pipeline import PipelineGraph
from vtkweb.execution import PipelineExecutionManager
from vtkweb.rendering import RenderManager
from vtkweb.state import export_python_state, load_python_state
from vtkweb.views import ViewManager
from vtkweb.workspace import WorkspaceManager
from vtkweb.distributed import context as distributed


def initialize_app_controller(
    server,
    pipeline: PipelineGraph,
    rendering: RenderManager,
    views: ViewManager,
    workspace: WorkspaceManager,
    catalog: AlgorithmCatalog,
) -> None:
    state = server.state
    ctrl = server.controller
    execution = PipelineExecutionManager(state, pipeline, rendering)
    execution_task: asyncio.Task | None = None

    # Server -> client request token for transient remote viewport dimensions.
    # The dimensions themselves remain outside serialized application state.
    state.remote_render_size_request_epoch = 0

    def request_render_sizes() -> None:
        state.remote_render_size_request_epoch = (
            int(state.remote_render_size_request_epoch or 0) + 1
        )

    # -------------------------------------------------------------------------
    # Primitive application commands
    # -------------------------------------------------------------------------

    def create_node(
        class_name: str,
        *,
        name: str | None = None,
        node_id: str | None = None,
    ) -> str:
        node_id = node_id or uuid4().hex
        distributed.replicate("create_node", class_name, name=name, node_id=node_id)
        descriptor = next(
            item for item in catalog.algorithms if item.class_name == class_name
        )
        processor = catalog.create(class_name)
        node = pipeline.add_node(
            processor,
            name=name,
            node_id=node_id,
        )
        return node.id

    def connect_nodes(
        source_node_id: str,
        target_node_id: str,
        *,
        source_port: int = 0,
        target_port: int = 0,
    ) -> None:
        distributed.replicate(
            "connect_nodes",
            source_node_id,
            target_node_id,
            source_port=int(source_port),
            target_port=int(target_port),
        )
        pipeline.connect(
            source_node_id,
            target_node_id,
            source_port=int(source_port),
            target_port=int(target_port),
        )

    def set_node_property(
        node_id: str,
        name: str,
        value,
    ) -> None:
        if state.pipeline_executing:
            return
        distributed.replicate("set_node_property", node_id, name, value)
        pipeline.set_property(
            node_id,
            name,
            value,
        )

    def set_node_input_array(
        node_id: str,
        index: int,
        value,
    ) -> None:
        if state.pipeline_executing:
            return
        distributed.replicate("set_node_input_array", node_id, int(index), value)
        pipeline.set_input_array(
            node_id,
            int(index),
            value,
        )

    def add_representation(
        node_id: str,
        output_port: int = 0,
        kind: str = "surface",
        view_ids: Iterable[str] = (),
        camera_reset_mode: int = 0,
        representation_id: str | None = None,
    ) -> str:
        representation_id = representation_id or uuid4().hex
        view_ids = tuple(view_ids)
        distributed.replicate(
            "add_representation",
            node_id,
            int(output_port),
            kind,
            view_ids,
            int(camera_reset_mode),
            representation_id,
        )
        return rendering.add_representation(
            node_id,
            output_port=int(output_port),
            kind=kind,
            view_ids=view_ids,
            camera_reset_mode=int(camera_reset_mode),
            representation_id=representation_id,
        ).id

    def toggle_representation_in_view(
        representation_id: str,
        view_id: str,
    ) -> None:
        distributed.replicate(
            "toggle_representation_in_view", representation_id, view_id
        )
        if rendering.representation_in_view(
            representation_id,
            view_id,
        ):
            rendering.unassign_representation(
                representation_id,
                view_id,
            )
        else:
            rendering.assign_representation(
                representation_id,
                view_id,
            )

    def create_view(
        view_type: str,
        *,
        name: str | None = None,
        view_id: str | None = None,
        **kwargs,
    ) -> str:
        view_id = view_id or uuid4().hex
        distributed.replicate(
            "create_view", view_type, name=name, view_id=view_id, **kwargs
        )
        return views.create_view(
            view_type,
            name=name,
            view_id=view_id,
            **kwargs,
        )

    def switch_view_type(view_id: str, view_type: str) -> None:
        distributed.replicate("switch_view_type", view_id, view_type)
        rendering.switch_view_type(view_id, view_type)
        request_render_sizes()

    def create_view_in_container(container_id: str, view_type: str) -> str:
        """Create a selected view backend in an empty workspace tile."""
        if workspace.state.workspace_nodes[container_id].get("view_id") is not None:
            raise ValueError(f"Container already has a view: {container_id}")
        view_id = create_view(view_type)
        assign_view_to_container(container_id, view_id)
        rendering.set_active_view(view_id)
        return view_id

    def remove_view(view_id: str) -> None:
        distributed.replicate("remove_view", view_id)
        workspace.close_view_tile(view_id)
        views.remove_view(view_id)

    def create_workspace(*, container_id: str | None = None) -> str:
        return workspace.create_workspace(container_id=container_id)

    def split_container(
        container_id: str,
        orientation: str,
        ratio: float = 0.5,
        *,
        first_id: str | None = None,
        second_id: str | None = None,
    ) -> tuple[str, str]:
        return workspace.split_container(
            container_id,
            orientation,
            ratio=float(ratio),
            first_id=first_id,
            second_id=second_id,
        )

    def assign_view_to_container(container_id: str, view_id: str | None) -> None:
        if view_id is not None:
            views.get(view_id)
            previous = workspace.container_for_view(view_id)
            if previous is not None and previous != container_id:
                workspace.assign_view(previous, None)
        workspace.assign_view(container_id, view_id)

    def set_split_ratio(container_id: str, ratio: float) -> None:
        workspace.set_split_ratio(container_id, float(ratio))

    def split_view_container(
        container_id: str,
        orientation: str,
    ) -> tuple[str, str]:
        """Split a leaf; the new leaf stays empty until a backend is selected."""
        return split_container(container_id, orientation)

    def set_active_node(
        node_id: str,
    ) -> None:
        distributed.replicate("set_active_node", node_id)
        pipeline.set_active_node(node_id)
        state.active_representation_output_port = 0

    # -------------------------------------------------------------------------
    # UI workflows
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

        representations = rendering.get_representations(
            node_id,
            output_port,
        )
        view_id = state.active_view_id

        if not representations:
            add_representation(
                node_id,
                output_port=output_port,
                kind="surface",
                view_ids=[view_id],
                camera_reset_mode=1,
            )
            return

        visible = any(
            view_id in representation.view_ids for representation in representations
        )

        for representation in representations:
            is_visible = view_id in representation.view_ids
            if visible and is_visible:
                distributed.replicate(
                    "toggle_representation_in_view",
                    representation.id,
                    view_id,
                )
                rendering.unassign_representation(
                    representation.id,
                    view_id,
                )
            elif not visible and not is_visible:
                distributed.replicate(
                    "toggle_representation_in_view",
                    representation.id,
                    view_id,
                )
                rendering.assign_representation(
                    representation.id,
                    view_id,
                )

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

        # Representations are intentionally not created at node insertion time.
        # The execution manager creates an outline for each output port after
        # the node completes its first successful execution.

        ctrl.close_node_browser()

        set_active_node(node_id)

    def delete_node(
        node_id: str,
    ) -> None:
        if node_id not in pipeline.nodes:
            return
        distributed.replicate("delete_node", node_id)

        rendering.remove_node(node_id)
        pipeline.remove_node(node_id)

        if pipeline.active_node is not None:
            set_active_node(pipeline.active_node.id)
        else:
            state.active_representation_output_port = 0

    def delete_active_node() -> None:
        node_id = pipeline.active_node_id
        if node_id is not None:
            delete_node(node_id)

    # -------------------------------------------------------------------------
    # Python state reconstruction
    # -------------------------------------------------------------------------

    def clear_state() -> None:
        """Clear all reconstructable application and workspace state."""
        distributed.replicate("clear_state")

        for representation in tuple(rendering.representations):
            rendering.remove_representation(representation.id)

        for view in tuple(views.views):
            # State files commonly recreate the same logical view IDs. Preserve
            # their client-reported viewport sizes across this reconstruction so
            # the replacement backend can render immediately even when the DOM
            # tile itself never resizes.
            views.remove_view(view["id"], preserve_runtime=True)

        workspace.clear()
        pipeline.clear()
        rendering.transfer_functions.clear()
        state.active_view_id = None
        state.active_representation_output_port = 0

    def finish_state_load() -> None:
        """Finalize reconstructed state without executing the pipeline.

        Loading restores configuration only. Pipeline execution remains an
        explicit user action. The browser may keep the same render DOM nodes
        across reconstruction, so ask it to re-report the actual viewport
        dimensions even when no ResizeObserver event occurs.
        """

        for node_id in pipeline.nodes:
            pipeline.mark_modified(node_id, include_downstream=False)
            pipeline.refresh_runtime_metadata(node_id)

        rendering.prune_render_sizes()
        request_render_sizes()

    def _start_execution_plan(node_order) -> None:
        nonlocal execution_task
        if execution_task is not None and not execution_task.done():
            return
        execution_task = asyncio.create_task(execution.execute_plan(tuple(node_order)))

        def consume_result(task: asyncio.Task) -> None:
            nonlocal execution_task
            execution_task = None
            try:
                task.result()
            except Exception as exc:
                print(f"Pipeline execution failed: {exc}")

        execution_task.add_done_callback(consume_result)

    def execute_pipeline() -> None:
        if execution_task is not None and not execution_task.done():
            return
        node_order = execution.execution_plan()
        distributed.replicate("execute_pipeline", tuple(node_order))
        _start_execution_plan(node_order)

    def abort_pipeline() -> None:
        execution.abort()

    def export_state_source() -> str:
        return export_python_state(pipeline, rendering, views, workspace)

    def load_state_source(
        source: str | bytes,
        *,
        filename: str = "<vtkweb-state>",
    ) -> None:
        load_python_state(
            source,
            ctrl,
            filename=filename,
        )

    def save_python_state_file(filename: str) -> str:
        """Save the current state to an explicit server-side path."""

        path = Path(filename).expanduser().resolve()
        path.write_text(
            export_state_source(),
            encoding="utf-8",
        )
        return str(path)

    def open_python_state_file(filename: str) -> str:
        """Load a trusted Python state file from an explicit server path."""

        path = Path(filename).expanduser().resolve()
        load_state_source(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
        return str(path)

    # MPI-replicated rendering mutations.  These wrappers keep MPI details out
    # of the rendering/public APIs and preserve the same call path on workers.
    def remove_representation(representation_id: str) -> None:
        distributed.replicate("remove_representation", representation_id)
        rendering.remove_representation(representation_id)

    def set_representation_kind(representation_id: str, kind: str) -> None:
        distributed.replicate("set_representation_kind", representation_id, kind)
        rendering.set_representation_kind(representation_id, kind)

    def set_representation_property(representation_id: str, name: str, value) -> None:
        distributed.replicate(
            "set_representation_property", representation_id, name, value
        )
        rendering.set_representation_property(representation_id, name, value)

    def set_active_view(view_id: str) -> None:
        distributed.replicate("set_active_view", view_id)
        rendering.set_active_view(view_id)

    def set_view_property(view_id: str, name: str, value) -> None:
        distributed.replicate("set_view_property", view_id, name, value)
        rendering.set_view_property(view_id, name, value)

    def reset_camera(view_id: str | None = None) -> None:
        if view_id is None:
            view_id = rendering.active_view_id
        if view_id is None:
            return
        # Reset on rank 0, then replicate the resulting renderer-agnostic camera
        # value. State-only worker views have no backend to compute bounds from.
        if distributed.is_root:
            rendering.reset_camera(view_id)
            distributed.replicate(
                "set_view_property",
                view_id,
                "camera",
                rendering.get_view_property(view_id, "camera"),
            )
        else:
            rendering.reset_camera(view_id)

    def interact_view_camera(view_id, mode, dx, dy, viewport_height) -> None:
        distributed.replicate(
            "interact_view_camera", view_id, mode, dx, dy, viewport_height
        )
        rendering.interact_view_camera(view_id, mode, dx, dy, viewport_height)

    def set_render_size(view_id: str, width: int, height: int) -> None:
        distributed.replicate("set_render_size", view_id, int(width), int(height))
        rendering.set_render_size(view_id, width, height)

    def set_tf_data(array_name, value) -> None:
        distributed.replicate("set_tf_data", array_name, value)
        rendering.transfer_functions.set_data(array_name, value)

    def apply_tf_preset(array_name, preset_name) -> None:
        distributed.replicate("apply_tf_preset", array_name, preset_name)
        rendering.transfer_functions.apply_preset(array_name, preset_name)

    def set_tf_range(array_name, minimum, maximum) -> None:
        distributed.replicate("set_tf_range", array_name, minimum, maximum)
        rendering.transfer_functions.set_range(array_name, minimum, maximum)

    def rescale_tf(array_name) -> None:
        distributed.replicate("rescale_tf", array_name)
        rendering.transfer_functions.rescale(array_name)

    def set_tf_control_point_component(
        array_name, point_index, component_index, value
    ) -> None:
        distributed.replicate(
            "set_tf_control_point_component",
            array_name,
            point_index,
            component_index,
            value,
        )
        rendering.transfer_functions.set_control_point_component(
            array_name, point_index, component_index, value
        )

    def add_tf_control_point(array_name) -> None:
        distributed.replicate("add_tf_control_point", array_name)
        rendering.transfer_functions.add_control_point(array_name)

    def remove_tf_control_point(array_name, point_index) -> None:
        distributed.replicate("remove_tf_control_point", array_name, point_index)
        rendering.transfer_functions.remove_control_point(array_name, point_index)

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    ctrl.create_node = create_node
    ctrl.connect_nodes = connect_nodes
    ctrl.set_node_property = set_node_property
    ctrl.set_node_input_array = set_node_input_array
    ctrl.add_representation = add_representation
    ctrl.remove_representation = remove_representation
    ctrl.set_representation_kind = set_representation_kind
    ctrl.toggle_representation_in_view = toggle_representation_in_view
    ctrl.set_representation_property = set_representation_property
    ctrl.set_tf_data = set_tf_data
    ctrl.apply_tf_preset = apply_tf_preset
    ctrl.set_tf_range = set_tf_range
    ctrl.rescale_tf = rescale_tf
    ctrl.set_tf_control_point_component = set_tf_control_point_component
    ctrl.add_tf_control_point = add_tf_control_point
    ctrl.remove_tf_control_point = remove_tf_control_point
    ctrl.create_view = create_view
    ctrl.create_view_in_container = create_view_in_container
    ctrl.remove_view = remove_view
    ctrl.switch_view_type = switch_view_type
    ctrl.create_workspace = create_workspace
    ctrl.split_container = split_container
    ctrl.assign_view_to_container = assign_view_to_container
    ctrl.split_view_container = split_view_container
    ctrl.set_active_view = set_active_view
    ctrl.set_view_property = set_view_property
    ctrl.reset_camera = reset_camera
    ctrl.set_active_node = set_active_node
    ctrl.output_port_click = output_port_click
    ctrl.insert_node = insert_node
    ctrl.delete_node = delete_node
    ctrl.clear_state = clear_state
    ctrl.finish_state_load = finish_state_load
    ctrl.request_render_sizes = request_render_sizes
    ctrl.export_python_state = export_state_source
    ctrl.load_python_state = load_state_source
    ctrl.save_python_state_file = save_python_state_file
    ctrl.open_python_state_file = open_python_state_file
    ctrl.execute_pipeline = execute_pipeline
    ctrl.abort_pipeline = abort_pipeline

    # Client-to-server render/workspace RPCs. UI code emits these events but
    # application/controller ownership stays here.
    ctrl.trigger("interact_view_camera")(interact_view_camera)
    ctrl.trigger("set_render_size")(set_render_size)
    ctrl.trigger("set_split_ratio")(set_split_ratio)

    distributed.register("create_node", create_node)
    distributed.register("connect_nodes", connect_nodes)
    distributed.register("set_node_property", set_node_property)
    distributed.register("set_node_input_array", set_node_input_array)
    distributed.register("add_representation", add_representation)
    distributed.register("remove_representation", remove_representation)
    distributed.register("set_representation_kind", set_representation_kind)
    distributed.register("set_representation_property", set_representation_property)
    distributed.register("toggle_representation_in_view", toggle_representation_in_view)
    distributed.register("create_view", create_view)
    distributed.register("remove_view", remove_view)
    distributed.register("switch_view_type", switch_view_type)
    distributed.register("set_active_view", set_active_view)
    distributed.register("set_view_property", set_view_property)
    distributed.register("reset_camera", reset_camera)
    distributed.register("interact_view_camera", interact_view_camera)
    distributed.register("set_render_size", set_render_size)
    distributed.register("set_tf_data", set_tf_data)
    distributed.register("apply_tf_preset", apply_tf_preset)
    distributed.register("set_tf_range", set_tf_range)
    distributed.register("rescale_tf", rescale_tf)
    distributed.register(
        "set_tf_control_point_component", set_tf_control_point_component
    )
    distributed.register("add_tf_control_point", add_tf_control_point)
    distributed.register("remove_tf_control_point", remove_tf_control_point)
    distributed.register("set_active_node", set_active_node)
    distributed.register("delete_node", delete_node)
    distributed.register("clear_state", clear_state)
    distributed.register("execute_pipeline", _start_execution_plan)

    server.trigger("delete_active_node")(delete_active_node)
    server.trigger("execute_pipeline")(execute_pipeline)
