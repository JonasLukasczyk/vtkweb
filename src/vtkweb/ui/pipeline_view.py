from __future__ import annotations

import asyncio
import shlex

import graphviz

from trame.widgets import flow, html
from trame.widgets import vuetify3 as v3

from vtkweb.pipeline import (
    PipelineEdge,
    PipelineGraph,
    PipelineNode,
)


PORT_SLOT_COUNT = 16


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

    display: inline-flex;
    align-items: center;
    justify-content: center;

    overflow: visible !important;

    min-width: 0;
    width: fit-content;
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
    pipeline: PipelineGraph,
):
    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------

    with v3.VCol(
        cols=3,
        classes="pa-2",
        style="height:100vh;",
    ):
        v3.VLabel(
            "Pipeline"
        )

        with v3.VCard(
            classes="mt-2",
            style=(
                "height:calc(100vh - 60px);"
            ),
        ):
            with flow.NodeEditor(
                style=(
                    "height:100%;"
                    "width:100%;"
                ),
            ) as node_editor:
                flow.Background()

                # -------------------------------------------------------------
                # Controls
                # -------------------------------------------------------------

                with flow.Controls(
                    classes="vtkweb-flow-controls",
                ):
                    with flow.ControlsButton(
                        title="Compute layout",
                        click=(
                            ctrl.compute_pipeline_layout
                        ),
                    ):
                        html.Div(
                            "↕",
                            style=(
                                "display:flex;"
                                "align-items:center;"
                                "justify-content:center;"
                                "width:100%;"
                                "height:100%;"
                                "font-size:14px;"
                                "cursor:pointer;"
                                "background:#fff;"
                            ),
                        )

                # -------------------------------------------------------------
                # Custom node
                # -------------------------------------------------------------

                with flow.CustomNode(
                    type="vtk-node",
                    var_name="node",
                ):
                    with v3.VCard(
                        classes=(
                            "pa-1 "
                            "vtkweb-pipeline-node"
                        ),
                        click=(
                            ctrl.set_active_node,
                            "[node.id]",
                        ),
                    ):
                        # -----------------------------------------------------
                        # Input handles
                        # -----------------------------------------------------

                        for port in range(
                            PORT_SLOT_COUNT
                        ):
                            flow.Handle(
                                id=f"input-{port}",
                                type="target",
                                position="top",
                                v_if=(
                                    f"node.data.input_port_count > {port}"
                                ),
                                classes=(
                                    "vtkweb-pipeline-handle "
                                    "vtkweb-input-handle"
                                ),
                                style=(
                                    "{ "
                                    "'top': '0px', "
                                    "'left': "
                                    f"((({port} + 1) / "
                                    "(node.data.input_port_count + 1)) "
                                    "* 100) + '%' "
                                    "}"
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
                            style=(
                                "white-space:nowrap;"
                            ),
                        )

                        # -----------------------------------------------------
                        # Output handles
                        # -----------------------------------------------------

                        for port in range(
                            PORT_SLOT_COUNT
                        ):
                            flow.Handle(
                                id=f"output-{port}",
                                type="source",
                                position="bottom",
                                v_if=(
                                    f"node.data.output_port_count > {port}"
                                ),
                                classes=(
                                    "'vtkweb-pipeline-handle ' + "
                                    "("
                                    "output_port_visibility[node.id] "
                                    "&& "
                                    f"output_port_visibility[node.id][{port}] "
                                    "? "
                                    "'vtkweb-output-visible' "
                                    ": "
                                    "'vtkweb-output-hidden'"
                                    ")"
                                ),
                                style=(
                                    "{ "
                                    "'bottom': '0px', "
                                    "'left': "
                                    f"((({port} + 1) / "
                                    "(node.data.output_port_count + 1)) "
                                    "* 100) + '%' "
                                    "}"
                                ),
                                click=(
                                    ctrl.output_port_click,
                                    (
                                        "["
                                        "node.id,"
                                        f"{port},"
                                        "$event.shiftKey"
                                        "]"
                                    ),
                                ),
                            )

    # -------------------------------------------------------------------------
    # Node serialization
    # -------------------------------------------------------------------------

    def node_data(
        node: PipelineNode,
    ) -> dict:
        index = list(
            pipeline.nodes
        ).index(
            node.id
        )

        algorithm = (
            node.algorithm
        )

        input_port_count = (
            algorithm.GetNumberOfInputPorts()
        )

        output_port_count = (
            algorithm.GetNumberOfOutputPorts()
        )

        if (
            input_port_count
            > PORT_SLOT_COUNT
        ):
            print(
                f"Warning: {node.name} has "
                f"{input_port_count} input ports, "
                f"but pipeline_view only provides "
                f"{PORT_SLOT_COUNT} handle slots."
            )

        if (
            output_port_count
            > PORT_SLOT_COUNT
        ):
            print(
                f"Warning: {node.name} has "
                f"{output_port_count} output ports, "
                f"but pipeline_view only provides "
                f"{PORT_SLOT_COUNT} handle slots."
            )

        return {
            "id": node.id,
            "type": "vtk-node",
            "label": node.name,

            "data": {
                "input_port_count": (
                    input_port_count
                ),
                "output_port_count": (
                    output_port_count
                ),
            },

            "position": {
                "x": 100,
                "y": (
                    80
                    + index * 140
                ),
            },
        }

    # -------------------------------------------------------------------------
    # Edge serialization
    # -------------------------------------------------------------------------

    def edge_data(
        edge: PipelineEdge,
    ) -> dict:
        return {
            "id": (
                f"{edge.source_node_id}-"
                f"{edge.source_port}-"
                f"{edge.target_node_id}-"
                f"{edge.target_port}"
            ),

            "source": (
                edge.source_node_id
            ),

            "target": (
                edge.target_node_id
            ),

            "sourceHandle": (
                f"output-{edge.source_port}"
            ),

            "targetHandle": (
                f"input-{edge.target_port}"
            ),
        }

    # -------------------------------------------------------------------------
    # Current positions
    # -------------------------------------------------------------------------

    def current_positions() -> dict[
        str,
        dict[str, float],
    ]:
        result = {}

        for node in (
            pipeline.nodes.values()
        ):
            editor_node = (
                node_editor.get_node(
                    node.id
                )
            )

            if editor_node is None:
                continue

            position = (
                editor_node.get(
                    "position"
                )
            )

            if position is None:
                continue

            result[node.id] = {
                "x": float(
                    position["x"]
                ),
                "y": float(
                    position["y"]
                ),
            }

        return result

    # -------------------------------------------------------------------------
    # Graphviz layout
    # -------------------------------------------------------------------------

    def compute_layout_positions() -> dict[
        str,
        dict[str, float],
    ]:
        graph = graphviz.Digraph(
            engine="dot"
        )

        for node in (
            pipeline.nodes.values()
        ):
            graph.node(
                node.id,
                label=node.name,
            )

        for edge in (
            pipeline.edges
        ):
            graph.edge(
                edge.source_node_id,
                edge.target_node_id,
            )

        plain = (
            graph.pipe(
                format="plain"
            )
            .decode(
                "utf-8"
            )
        )

        lines = (
            plain.splitlines()
        )

        graph_height = 0.0

        if lines:
            fields = (
                shlex.split(
                    lines[0]
                )
            )

            if (
                fields
                and fields[0]
                == "graph"
            ):
                graph_height = (
                    float(
                        fields[3]
                    )
                )

        scale = 120.0
        padding = 40.0

        positions = {}

        for line in lines:
            fields = (
                shlex.split(
                    line
                )
            )

            if (
                not fields
                or fields[0]
                != "node"
            ):
                continue

            node_id = (
                fields[1]
            )

            x = float(
                fields[2]
            )

            y = float(
                fields[3]
            )

            positions[
                node_id
            ] = {
                "x": (
                    padding
                    + x * scale
                ),
                "y": (
                    padding
                    + (
                        graph_height
                        - y
                    )
                    * scale
                ),
            }

        return positions

    # -------------------------------------------------------------------------
    # Layout animation
    # -------------------------------------------------------------------------

    async def animate_positions(
        start_positions: dict[
            str,
            dict[str, float],
        ],
        end_positions: dict[
            str,
            dict[str, float],
        ],
        duration: float = 0.2,
    ) -> None:
        fps = 60.0

        steps = max(
            1,
            round(
                duration
                * fps
            ),
        )

        for step in range(
            1,
            steps + 1,
        ):
            t = (
                step
                / steps
            )

            # Smoothstep easing
            alpha = (
                t
                * t
                * (
                    3.0
                    - 2.0 * t
                )
            )

            for (
                node_id,
                end,
            ) in (
                end_positions.items()
            ):
                start = (
                    start_positions.get(
                        node_id
                    )
                )

                if start is None:
                    continue

                position = {
                    "x": (
                        start["x"]
                        + (
                            end["x"]
                            - start["x"]
                        )
                        * alpha
                    ),
                    "y": (
                        start["y"]
                        + (
                            end["y"]
                            - start["y"]
                        )
                        * alpha
                    ),
                }

                node_editor.update_node(
                    node_id,
                    position=position,
                )

            await asyncio.sleep(
                1.0 / fps
            )

        # Land exactly on the final
        # Graphviz coordinates.
        for (
            node_id,
            position,
        ) in (
            end_positions.items()
        ):
            node_editor.update_node(
                node_id,
                position=position,
            )

    def compute_pipeline_layout() -> None:
        start_positions = (
            current_positions()
        )

        end_positions = (
            compute_layout_positions()
        )

        asyncio.create_task(
            animate_positions(
                start_positions,
                end_positions,
            )
        )

    ctrl.compute_pipeline_layout = (
        compute_pipeline_layout
    )

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------

    def fit_view() -> None:
        node_editor.fit_view()

    ctrl.pipeline_fit_view = (
        fit_view
    )

    # -------------------------------------------------------------------------
    # Remove
    # -------------------------------------------------------------------------

    def remove_node(
        node_id: str,
    ) -> None:
        node_editor.remove_node(
            node_id
        )

    def remove_edge(
        edge: PipelineEdge,
    ) -> None:
        node_editor.remove_edge(
            edge.source_node_id,
            edge.target_node_id,
            source_handle=(
                f"output-{edge.source_port}"
            ),
            target_handle=(
                f"input-{edge.target_port}"
            ),
        )

    ctrl.pipeline_remove_node = (
        remove_node
    )

    ctrl.pipeline_remove_edge = (
        remove_edge
    )

    # -------------------------------------------------------------------------
    # Add
    # -------------------------------------------------------------------------

    def add_node(
        node: PipelineNode,
    ) -> None:
        node_editor.add_node(
            node_data(
                node
            )
        )

    def add_edge(
        edge: PipelineEdge,
    ) -> None:
        async def deferred_add():
            # Give Vue Flow time to instantiate
            # and measure the custom node handles.
            await asyncio.sleep(
                0.05
            )

            node_editor.add_edge(
                edge_data(
                    edge
                )
            )

            node_editor.fit_view()

        asyncio.create_task(
            deferred_add()
        )

    ctrl.pipeline_add_node = (
        add_node
    )

    ctrl.pipeline_add_edge = (
        add_edge
    )

    # -------------------------------------------------------------------------
    # Initial graph
    # -------------------------------------------------------------------------

    for node in (
        pipeline.nodes.values()
    ):
        node_editor.add_node(
            node_data(
                node
            )
        )

    def initialize_edges():
        for edge in (
            pipeline.edges
        ):
            node_editor.add_edge(
                edge_data(
                    edge
                )
            )

        node_editor.fit_view()

    ctrl.on_client_connected.add(
        initialize_edges
    )

    return node_editor
