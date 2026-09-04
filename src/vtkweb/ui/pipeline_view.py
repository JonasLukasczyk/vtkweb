from __future__ import annotations

import asyncio
import shlex

import graphviz

from trame.widgets import flow, html
from trame.widgets import vuetify3 as v3


PIPELINE_VIEW_STYLE = """
.vtkweb-flow-controls {
    top: 10px !important;
    left: 10px !important;
    bottom: auto !important;
    right: auto !important;
}

.vtkweb-flow-controls .vue-flow__controls-button {
    cursor: pointer;
}

.vtkweb-pipeline-node {
    position: relative;
    background: #434343;

    display: inline-flex;
    align-items: center;
    justify-content: center;

    overflow: visible !important;

    min-width: 0;
    width: fit-content;
}

.vtkweb-pipeline-node-active {
    background: #125288 !important;
}

.vue-flow__handle.vtkweb-pipeline-handle {
    width: 16px !important;
    height: 16px !important;

    min-width: 16px !important;
    min-height: 16px !important;

    border: 2px solid white !important;
    border-radius: 50% !important;

    cursor: pointer !important;

    z-index: 20;
}

.vue-flow__handle.vtkweb-input-handle {
    background: #777 !important;
}

.vue-flow__handle.vtkweb-output-visible {
    background: #4caf50 !important;
}

.vue-flow__handle.vtkweb-output-hidden {
    background: #f44336 !important;
}

.vue-flow__handle.vtkweb-pipeline-handle:hover {
    filter: brightness(1.25);
}
"""


def build_pipeline_view(
    state,
    ctrl,
):
    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------

    with v3.VCol(
        cols=3,
        classes="pa-2",
        style="height:100vh;",
    ):
        with v3.VCard(
            classes="mt-2",
            style=("height:calc(100vh - 60px);"),
        ):
            with flow.NodeEditor(
                style=("height:100%;width:100%;"),
                node_origin=("[0.5, 0.5]",),
            ) as node_editor:
                flow.Background()

                # -------------------------------------------------------------
                # Custom node
                # -------------------------------------------------------------

                with flow.CustomNode(
                    type="vtk-node",
                    var_name="node",
                ):
                    with v3.VCard(
                        classes=(
                            (
                                "'pa-1 vtkweb-pipeline-node ' + "
                                "("
                                "node.id === active_node_id "
                                "? 'vtkweb-pipeline-node-active' "
                                ": ''"
                                ")"
                            ),
                        ),
                        click=(
                            ctrl.set_active_node,
                            "[node.id]",
                        ),
                    ):
                        # -----------------------------------------------------
                        # Input handles
                        # -----------------------------------------------------

                        flow.Handle(
                            id=("`input-${port - 1}`",),
                            key=("`input-${port - 1}`",),
                            type="target",
                            position="top",
                            v_for="port in node.data.input_port_count",
                            classes=("vtkweb-pipeline-handle vtkweb-input-handle"),
                            style=(
                                "{ "
                                "'top': '0px', "
                                "'left': "
                                "((port / "
                                "(node.data.input_port_count + 1)) "
                                "* 100) + '%' "
                                "}",
                            ),
                            click=(
                                ctrl.set_active_node,
                                "[node.id]",
                            ),
                        )

                        # -----------------------------------------------------
                        # Label
                        # -----------------------------------------------------

                        v3.VCardText(
                            "{{ node.label }}",
                            classes="pa-1",
                            style=("white-space:nowrap;font-family:monospace;"),
                        )

                        # -----------------------------------------------------
                        # Output handles
                        # -----------------------------------------------------

                        flow.Handle(
                            id=("`output-${port - 1}`",),
                            key=("`output-${port - 1}`",),
                            type="source",
                            position="bottom",
                            v_for="port in node.data.output_port_count",
                            classes=(
                                (
                                    "'vtkweb-pipeline-handle ' + "
                                    "("
                                    "Object.values(representations).some("
                                    "rep => "
                                    "rep.node_id === node.id && "
                                    "rep.output_port === port - 1 && "
                                    "rep.view_ids.includes(active_view_id)"
                                    ") "
                                    "? 'vtkweb-output-visible' "
                                    ": 'vtkweb-output-hidden'"
                                    ")"
                                ),
                            ),
                            style=(
                                "{ "
                                "'bottom': '0px', "
                                "'left': "
                                "((port / "
                                "(node.data.output_port_count + 1)) "
                                "* 100) + '%' "
                                "}",
                            ),
                            click=(
                                ctrl.output_port_click,
                                "[node.id, port - 1, $event.shiftKey]",
                            ),
                        )

    # -------------------------------------------------------------------------
    # Serialization from authoritative state
    # -------------------------------------------------------------------------

    def node_data(
        node_id: str,
        value: dict,
        index: int,
    ) -> dict:
        return {
            "id": node_id,
            "type": "vtk-node",
            "label": value["name"],
            "data": {
                "input_port_count": int(value["input_port_count"]),
                "output_port_count": int(value["output_port_count"]),
            },
            "position": {
                "x": 100,
                "y": 80 + index * 140,
            },
        }

    def edge_key(value: dict) -> tuple[str, int, str, int]:
        return (
            value["source_node_id"],
            int(value["source_port"]),
            value["target_node_id"],
            int(value["target_port"]),
        )

    def edge_data(value: dict) -> dict:
        source, source_port, target, target_port = edge_key(value)
        return {
            "id": f"{source}-{source_port}-{target}-{target_port}",
            "source": source,
            "target": target,
            "sourceHandle": f"output-{source_port}",
            "targetHandle": f"input-{target_port}",
        }

    def current_positions() -> dict[str, dict[str, float]]:
        result = {}
        for node_id in state.pipeline["nodes"]:
            editor_node = node_editor.get_node(node_id)
            if editor_node is None:
                continue
            position = editor_node.get("position")
            if position is None:
                continue
            result[node_id] = {
                "x": float(position["x"]),
                "y": float(position["y"]),
            }
        return result

    def compute_layout_positions() -> dict[str, dict[str, float]]:
        graph = graphviz.Digraph(engine="dot")
        scale = 40.0
        padding = 20.0
        char_width_px = 8.5
        horizontal_padding_px = 24.0
        node_height_px = 40.0
        min_width_px = 80.0
        node_sizes = {}

        for node_id, node in state.pipeline["nodes"].items():
            width_px = max(
                min_width_px,
                len(node["name"]) * char_width_px + horizontal_padding_px,
            )
            node_sizes[node_id] = {
                "width": width_px,
                "height": node_height_px,
            }
            graph.node(
                node_id,
                label="",
                width=str(width_px / scale),
                height=str(node_height_px / scale),
                fixedsize="true",
            )

        for edge in state.pipeline["edges"]:
            graph.edge(
                edge["source_node_id"],
                edge["target_node_id"],
            )

        plain = graph.pipe(format="plain").decode("utf-8")
        lines = plain.splitlines()
        graph_height = 0.0
        if lines:
            fields = shlex.split(lines[0])
            if fields and fields[0] == "graph":
                graph_height = float(fields[3])

        positions = {}
        for line in lines:
            fields = shlex.split(line)
            if not fields or fields[0] != "node":
                continue
            node_id = fields[1]
            center_x = float(fields[2])
            center_y = float(fields[3])
            size = node_sizes[node_id]
            positions[node_id] = {
                "x": padding + center_x * scale - size["width"] * 0.5,
                "y": padding + (graph_height - center_y) * scale - size["height"] * 0.5,
            }
        return positions

    async def animate_positions(
        start_positions: dict[str, dict[str, float]],
        end_positions: dict[str, dict[str, float]],
        duration: float = 0.2,
    ) -> None:
        steps = max(1, round(duration * 60.0))
        for step in range(1, steps + 1):
            t = step / steps
            alpha = t * t * (3.0 - 2.0 * t)
            for node_id, end in end_positions.items():
                start = start_positions.get(node_id)
                if start is None:
                    continue
                node_editor.update_node(
                    node_id,
                    position={
                        "x": start["x"] + (end["x"] - start["x"]) * alpha,
                        "y": start["y"] + (end["y"] - start["y"]) * alpha,
                    },
                )
            await asyncio.sleep(1.0 / 60.0)

        for node_id, position in end_positions.items():
            node_editor.update_node(node_id, position=position)

    sync_task: asyncio.Task | None = None
    known_nodes: set[str] = set()
    known_edges: set[tuple[str, int, str, int]] = set()

    def schedule_sync() -> None:
        nonlocal sync_task
        if sync_task is not None:
            sync_task.cancel()

        async def sync_view() -> None:
            nonlocal known_nodes, known_edges
            await asyncio.sleep(0.05)

            pipeline_state = state.pipeline
            nodes = pipeline_state["nodes"]
            edges = pipeline_state["edges"]
            current_nodes = set(nodes)
            current_edges = {edge_key(edge) for edge in edges}

            for source, source_port, target, target_port in known_edges - current_edges:
                node_editor.remove_edge(
                    source,
                    target,
                    source_handle=f"output-{source_port}",
                    target_handle=f"input-{target_port}",
                )

            for node_id in known_nodes - current_nodes:
                node_editor.remove_node(node_id)

            for index, (node_id, value) in enumerate(nodes.items()):
                if node_id not in known_nodes:
                    node_editor.add_node(node_data(node_id, value, index))

            if current_nodes - known_nodes:
                await asyncio.sleep(0.05)

            for edge in edges:
                if edge_key(edge) not in known_edges:
                    node_editor.add_edge(edge_data(edge))

            known_nodes = current_nodes
            known_edges = current_edges

            start_positions = current_positions()
            end_positions = compute_layout_positions()
            await animate_positions(start_positions, end_positions)

        sync_task = asyncio.create_task(sync_view())

    @state.change("pipeline")
    def on_pipeline_change(**_):
        schedule_sync()

    def initialize_graph() -> None:
        schedule_sync()

        async def fit_after_sync():
            await asyncio.sleep(0.35)
            node_editor.fit_view()

        asyncio.create_task(fit_after_sync())

    ctrl.on_client_connected.add(initialize_graph)

    return node_editor
