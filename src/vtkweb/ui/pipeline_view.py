from __future__ import annotations

import asyncio
import shlex

import graphviz

from trame.widgets import flow, html


NODE_HEIGHT = 32.0
NODE_MIN_WIDTH = 80.0
NODE_HORIZONTAL_PADDING = 12.0
NODE_BORDER_WIDTH = 2.0
NODE_CHAR_WIDTH = 8.5
GRAPHVIZ_SCALE = 40.0


PIPELINE_VIEW_STYLE = """
.vtkweb-pipeline-node {
    position: relative;

    display: inline-flex;
    align-items: end;
    justify-content: center;

    height: 32px;
    padding: 0 12px;

    box-sizing: border-box;
    overflow: visible !important;

    background: #313844;
    border: 2px solid transparent;
    border-radius: 4px;

    color: #f4f7fb;
    font-family: monospace;
    white-space: nowrap;

    cursor: pointer;
}

.vtkweb-pipeline-node-active {
    border-color: #58a6ff;
}

.vtkweb-pipeline-node-modified {
    background: #313844;
}

.vtkweb-pipeline-node-queued {
    background: #536273;
}

.vtkweb-pipeline-node-running {
    background-color: #536273;
    background-image: linear-gradient(
        110deg,
        #536273 0%,
        #536273 32%,
        #71849a 44%,
        #91a4b8 50%,
        #71849a 56%,
        #536273 68%,
        #536273 100%
    );
    background-repeat: no-repeat;
    background-size: 220% 100%;
    animation: vtkweb-running-gradient 1.1s linear infinite;
}

.vtkweb-pipeline-node-failed {
    background: #6b3942;
    color: #fff5f6;
}

.vtkweb-pipeline-node-success {
    background: #3f554d;
    color: #f4faf7;
}

@keyframes vtkweb-running-gradient {
    from {
        background-position: 100% 0;
    }

    to {
        background-position: -100% 0;
    }
}

@media (prefers-reduced-motion: reduce) {
    .vtkweb-pipeline-node-running {
        animation: none;
        background: #607184;
    }
}

.vue-flow__handle.vtkweb-pipeline-handle {
    width: 10px !important;
    height: 10px !important;

    min-width: 10px !important;
    min-height: 10px !important;

    border: 0 !important;
    border-radius: 50% !important;

    cursor: pointer !important;
    z-index: 20;
}

.vue-flow__handle.vtkweb-input-handle {
    top: -1px !important;
    background: #777 !important;
}

.vue-flow__handle.vtkweb-output-handle {
    bottom: -1px !important;
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


def build_pipeline_view(state, ctrl):
    node_editor = None

    def node_width(name: str) -> float:
        return max(
            NODE_MIN_WIDTH,
            len(name) * NODE_CHAR_WIDTH
            + 2 * NODE_HORIZONTAL_PADDING
            + 2 * NODE_BORDER_WIDTH,
        )

    def node_data(node_id: str, node: dict, index: int) -> dict:
        return {
            "id": node_id,
            "type": "vtk-node",
            "label": node["name"],
            "data": {
                "input_port_count": int(node["input_port_count"]),
                "output_port_count": int(node["output_port_count"]),
                "execution_state": node.get("execution_state", "modified"),
                "width": node_width(node["name"]),
            },
            "position": {
                "x": 100,
                "y": 80 + index * 140,
            },
        }

    def edge_key(edge: dict) -> tuple[str, int, str, int]:
        return (
            edge["source_node_id"],
            int(edge["source_port"]),
            edge["target_node_id"],
            int(edge["target_port"]),
        )

    def edge_data(edge: dict) -> dict:
        source, source_port, target, target_port = edge_key(edge)

        return {
            "id": f"{source}-{source_port}-{target}-{target_port}",
            "source": source,
            "target": target,
            "sourceHandle": f"output-{source_port}",
            "targetHandle": f"input-{target_port}",
        }

    def current_positions() -> dict[str, dict[str, float]]:
        positions = {}

        for node_id in state.pipeline["nodes"]:
            node = node_editor.get_node(node_id)

            if not node:
                continue

            position = node.get("position")

            if position is None:
                continue

            positions[node_id] = {
                "x": float(position["x"]),
                "y": float(position["y"]),
            }

        return positions

    def compute_layout_positions() -> dict[str, dict[str, float]]:
        graph = graphviz.Digraph(engine="dot")
        sizes = {}

        for node_id, node in state.pipeline["nodes"].items():
            width = node_width(node["name"])

            sizes[node_id] = {
                "width": width,
                "height": NODE_HEIGHT,
            }

            graph.node(
                node_id,
                label="",
                width=str(width / GRAPHVIZ_SCALE),
                height=str(NODE_HEIGHT / GRAPHVIZ_SCALE),
                fixedsize="true",
            )

        for edge in state.pipeline["edges"]:
            graph.edge(
                edge["source_node_id"],
                edge["target_node_id"],
            )

        lines = graph.pipe(format="plain").decode("utf-8").splitlines()

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

            center_x = float(fields[2]) * GRAPHVIZ_SCALE
            center_y = (graph_height - float(fields[3])) * GRAPHVIZ_SCALE

            size = sizes[node_id]

            positions[node_id] = {
                "x": center_x - size["width"] * 0.5,
                "y": center_y - size["height"] * 0.5,
            }

        return positions

    async def animate_positions(
        start_positions: dict[str, dict[str, float]],
        end_positions: dict[str, dict[str, float]],
        duration: float = 0.2,
    ) -> None:
        steps = max(1, round(duration * 60))

        for step in range(1, steps + 1):
            t = step / steps
            alpha = t * t * (3 - 2 * t)

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

            await asyncio.sleep(1 / 60)

        for node_id, position in end_positions.items():
            node_editor.update_node(
                node_id,
                position=position,
            )

    async def relayout() -> None:
        await animate_positions(
            current_positions(),
            compute_layout_positions(),
        )

    def pipeline_view_space() -> None:
        asyncio.create_task(relayout())

    ctrl.trigger("pipeline_view_space")(pipeline_view_space)

    with html.Div(
        style=("height:100%;width:100%;min-width:0;min-height:0;outline:none;"),
        tabindex=0,
        raw_attrs=[
            (
                '@keydown.space="'
                "['INPUT','TEXTAREA','SELECT','BUTTON'].includes("
                "$event.target.tagName"
                ") || ("
                "$event.preventDefault(), "
                "trigger('pipeline_view_space')"
                ")"
                '"'
            ),
        ],
    ):
        with flow.NodeEditor(
            style="height:100%;width:100%;",
        ) as node_editor:
            flow.Background()

            with flow.CustomNode(
                type="vtk-node",
                var_name="node",
            ):
                with html.Div(
                    "{{ node.label }}",
                    classes=(
                        (
                            "'vtkweb-pipeline-node ' + "
                            "'vtkweb-pipeline-node-' + "
                            "node.data.execution_state + ' ' + "
                            "("
                            "node.id === active_node_id "
                            "? 'vtkweb-pipeline-node-active' "
                            ": ''"
                            ")"
                        ),
                    ),
                    style=("{'width': node.data.width + 'px'}"),
                    click=(
                        ctrl.set_active_node,
                        "[node.id]",
                    ),
                ):
                    flow.Handle(
                        id=("`input-${port - 1}`",),
                        key=("`input-${port - 1}`",),
                        type="target",
                        position="top",
                        v_for="port in node.data.input_port_count",
                        classes=("vtkweb-pipeline-handle vtkweb-input-handle"),
                        style=(
                            "{"
                            "'left': "
                            "(port / "
                            "(node.data.input_port_count + 1) "
                            "* 100) + '%'"
                            "}"
                        ),
                        click=(
                            ctrl.set_active_node,
                            "[node.id]",
                        ),
                    )

                    flow.Handle(
                        id=("`output-${port - 1}`",),
                        key=("`output-${port - 1}`",),
                        type="source",
                        position="bottom",
                        v_for="port in node.data.output_port_count",
                        classes=(
                            (
                                "'vtkweb-pipeline-handle "
                                "vtkweb-output-handle ' + "
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
                            "{"
                            "'left': "
                            "(port / "
                            "(node.data.output_port_count + 1) "
                            "* 100) + '%'"
                            "}"
                        ),
                        click=(
                            ctrl.output_port_click,
                            "[node.id, port - 1, $event.shiftKey]",
                        ),
                    )

    sync_task: asyncio.Task | None = None
    sync_pending = False
    known_nodes: set[str] = set()
    known_edges: set[tuple[str, int, str, int]] = set()

    def schedule_sync() -> None:
        nonlocal sync_task, sync_pending

        sync_pending = True

        if sync_task is not None and not sync_task.done():
            return

        async def sync() -> None:
            nonlocal sync_task
            nonlocal sync_pending
            nonlocal known_nodes
            nonlocal known_edges

            try:
                while sync_pending:
                    sync_pending = False

                    await asyncio.sleep(0)

                    nodes = state.pipeline["nodes"]
                    edges = state.pipeline["edges"]

                    current_nodes = set(nodes)
                    current_edges = {edge_key(edge) for edge in edges}

                    topology_changed = (
                        current_nodes != known_nodes or current_edges != known_edges
                    )

                    for edge in known_edges - current_edges:
                        (
                            source,
                            source_port,
                            target,
                            target_port,
                        ) = edge

                        node_editor.remove_edge(
                            source,
                            target,
                            source_handle=f"output-{source_port}",
                            target_handle=f"input-{target_port}",
                        )

                    for node_id in known_nodes - current_nodes:
                        node_editor.remove_node(node_id)

                    for index, (node_id, node) in enumerate(nodes.items()):
                        serialized = node_data(
                            node_id,
                            node,
                            index,
                        )

                        if node_id not in known_nodes:
                            node_editor.add_node(serialized)

                        else:
                            node_editor.update_node(
                                node_id,
                                label=serialized["label"],
                                data=serialized["data"],
                            )

                    for edge in edges:
                        if edge_key(edge) not in known_edges:
                            node_editor.add_edge(edge_data(edge))

                    known_nodes = current_nodes
                    known_edges = current_edges

                    if topology_changed:
                        await asyncio.sleep(0)
                        await relayout()

            finally:
                sync_task = None

                if sync_pending:
                    schedule_sync()

        sync_task = asyncio.create_task(sync())

    @state.change("pipeline")
    def on_pipeline_change(**_):
        schedule_sync()

    def initialize_graph() -> None:
        schedule_sync()

        async def fit_after_sync() -> None:
            await asyncio.sleep(0.35)
            node_editor.fit_view()

        asyncio.create_task(fit_after_sync())

    ctrl.on_client_connected.add(initialize_graph)

    return node_editor
