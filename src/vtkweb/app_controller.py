from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from vtkweb.catalog import AlgorithmCatalog
from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering import RenderManager
from vtkweb.state import export_python_state, load_python_state


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
        return node.id

    def connect_nodes(
        source_node_id: str,
        target_node_id: str,
        *,
        source_port: int = 0,
        target_port: int = 0,
        sync: bool = True,
    ) -> None:
        pipeline.connect(
            source_node_id,
            target_node_id,
            source_port=int(source_port),
            target_port=int(target_port),
            sync=sync,
        )

    def set_node_property(
        node_id: str,
        name: str,
        value,
    ) -> None:
        pipeline.set_property(node_id, name, value)

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

    def add_node_list_value(
        node_id: str,
        name: str,
    ) -> None:
        pipeline.add_list_value(node_id, name)

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

    def sync_node_from_runtime(
        node_id: str,
    ) -> None:
        pipeline.sync_node_from_runtime(node_id)

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
        return representation.id

    def remove_representation(
        representation_id: str,
    ) -> None:
        rendering.remove_representation(representation_id)

    def set_representation_kind(
        representation_id: str,
        kind: str,
    ) -> None:
        rendering.set_representation_kind(
            representation_id,
            kind,
        )

    def toggle_representation_in_view(
        representation_id: str,
        view_id: str,
    ) -> None:
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

    def set_representation_array(
        representation_id: str,
        array_name: str | None,
        association: str = "point",
    ) -> None:
        rendering.set_array(
            representation_id,
            array_name,
            association,
        )

    def set_representation_color(
        representation_id: str,
        value: str,
    ) -> None:
        rendering.set_color(
            representation_id,
            value,
        )

    def set_representation_scalar_range(
        representation_id: str,
        minimum: float,
        maximum: float,
    ) -> None:
        rendering.set_scalar_range(
            representation_id,
            float(minimum),
            float(maximum),
        )

    def set_active_view(
        view_id: str,
    ) -> None:
        rendering.set_active_view(view_id)

    def set_view_background_color(
        view_id: str,
        value: str,
    ) -> None:
        rendering.set_background_color(
            view_id,
            _hex_to_rgb(value),
        )

    def reset_camera(
        view_id: str,
    ) -> None:
        rendering.reset_camera(view_id)

    def restore_view(
        *,
        name: str,
        view_id: str,
    ) -> str:
        """Restore the ID of the single render view without replacing it."""

        views = rendering.views
        if len(views) != 1:
            raise RuntimeError(
                "Current vtkweb UI can restore state only with one render view"
            )

        restored = rendering.rename_view(
            views[0].id,
            view_id,
            name=name,
        )
        rendering.set_active_view(restored.id)
        return restored.id

    def set_active_node(
        node_id: str,
    ) -> None:
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
            representation_id = add_representation(
                node_id,
                output_port=output_port,
                kind="surface",
            )
            rendering.assign_representation(
                representation_id,
                view_id,
            )
            return

        visible = any(
            view_id in representation.view_ids for representation in representations
        )

        for representation in representations:
            if visible:
                rendering.unassign_representation(
                    representation.id,
                    view_id,
                )
            else:
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

        if processor.GetNumberOfOutputPorts() > 0:
            add_representation(
                node_id,
                output_port=0,
                kind="outline",
                view_ids={state.active_view_id},
            )

        ctrl.close_node_browser()

        # Required automatic inputs are connected before processor inspection.
        processor.Update()
        pipeline.sync_node_from_runtime(node_id)
        set_active_node(node_id)
        rendering.reset_camera(state.active_view_id)

    def delete_node(
        node_id: str,
    ) -> None:
        if node_id not in pipeline.nodes:
            return

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
        """Clear reconstructable state while preserving the displayed view."""

        for representation in tuple(rendering.representations):
            rendering.remove_representation(representation.id)

        pipeline.clear()
        state.active_representation_output_port = 0

    def finish_state_load() -> None:
        """Synchronize runtime metadata after reconstruction is complete."""

        for node_id in pipeline.nodes:
            pipeline.sync_node_from_runtime(node_id)

        if rendering.active_view_id is not None:
            rendering.reset_camera(rendering.active_view_id)

    def export_state_source() -> str:
        return export_python_state(pipeline, rendering)

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

    def save_python_state() -> str | None:
        """Save the current state to a server-side Python file."""

        filename = _ask_save_state_filename()
        if filename is None:
            return None

        path = Path(filename)
        path.write_text(
            export_state_source(),
            encoding="utf-8",
        )
        return str(path)

    def open_python_state() -> str | None:
        """Load a trusted Python state file from the server filesystem."""

        filename = _ask_open_state_filename()
        if filename is None:
            return None

        path = Path(filename)
        load_state_source(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
        return str(path)

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
    ctrl.sync_node_from_runtime = sync_node_from_runtime
    ctrl.add_representation = add_representation
    ctrl.remove_representation = remove_representation
    ctrl.set_representation_kind = set_representation_kind
    ctrl.toggle_representation_in_view = toggle_representation_in_view
    ctrl.set_representation_array = set_representation_array
    ctrl.set_representation_color = set_representation_color
    ctrl.set_representation_scalar_range = set_representation_scalar_range
    ctrl.set_active_view = set_active_view
    ctrl.set_view_background_color = set_view_background_color
    ctrl.reset_camera = reset_camera
    ctrl.restore_view = restore_view
    ctrl.set_active_node = set_active_node
    ctrl.output_port_click = output_port_click
    ctrl.insert_node = insert_node
    ctrl.delete_node = delete_node
    ctrl.clear_state = clear_state
    ctrl.finish_state_load = finish_state_load
    ctrl.export_python_state = export_state_source
    ctrl.load_python_state = load_state_source
    ctrl.save_python_state = save_python_state
    ctrl.open_python_state = open_python_state

    server.trigger("delete_active_node")(delete_active_node)


def _ask_save_state_filename() -> str | None:
    """Open a server-side save-file dialog for a Python state file."""

    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    try:
        filename = filedialog.asksaveasfilename(
            title="Save vtkweb State",
            defaultextension=".py",
            filetypes=[
                ("Python state files", "*.py"),
                ("All files", "*"),
            ],
            initialfile="vtkweb_state.py",
        )
    finally:
        root.destroy()

    return filename or None


def _ask_open_state_filename() -> str | None:
    """Open a server-side file dialog for a trusted Python state file."""

    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    try:
        filename = filedialog.askopenfilename(
            title="Open vtkweb State",
            filetypes=[
                ("Python state files", "*.py"),
                ("All files", "*"),
            ],
        )
    finally:
        root.destroy()

    return filename or None


def _hex_to_rgb(
    value: str,
) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )
