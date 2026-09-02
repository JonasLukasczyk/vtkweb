from __future__ import annotations

import asyncio

from trame.widgets import flow, html
from trame.widgets import vuetify3 as v3

from vtkweb.pipeline import (
    PipelineEdge,
    PipelineGraph,
    PipelineNode,
)


def build_pipeline_view(
    ctrl,
    pipeline: PipelineGraph,
):
    with v3.VCol(
        cols=3,
        classes="pa-2",
        style="height:100vh;",
    ):
        v3.VLabel("Pipeline")

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
                flow.Controls()

                with flow.CustomNode(
                    type="vtk-node",
                    var_name="node",
                ):
                    with v3.VCard(
                        classes="pa-1",
                        style=(
                            "display:inline-flex;"
                            "flex-direction:column;"
                            "align-items:center;"
                            "overflow:visible;"
                            "min-width:0;"
                            "width:fit-content;"
                        ),
                        click=(
                            ctrl.node_click,
                            (
                                "[node.id, "
                                "$event.shiftKey]"
                            ),
                        ),
                    ):
                        flow.Handle(
                            type="target",
                            position="top",
                            style="top:-6px;",
                        )

                        with v3.VRow(
                            no_gutters=True,
                            align="center",
                            classes="px-1",
                        ):
                            v3.VCardText(
                                "{{ node.label }}",
                                classes="pa-1",
                                style=(
                                    "white-space:"
                                    "nowrap;"
                                ),
                            )

                            html.Span(
                                "👁",
                                v_if=(
                                    "node_visibility"
                                    "[node.id]"
                                ),
                                classes="ml-1",
                                style=(
                                    "font-size:14px;"
                                    "line-height:1;"
                                    "user-select:none;"
                                ),
                            )

                        flow.Handle(
                            type="source",
                            position="bottom",
                            style="bottom:-6px;",
                        )

    def node_data(
        node: PipelineNode,
    ) -> dict:
        index = list(
            pipeline.nodes
        ).index(node.id)

        return {
            "id": node.id,
            "type": "vtk-node",
            "label": node.name,
            "position": {
                "x": 100,
                "y": 80 + index * 140,
            },
        }

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
        }

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
        )

    ctrl.pipeline_remove_node = remove_node
    ctrl.pipeline_remove_edge = remove_edge

    def add_node(
        node: PipelineNode,
    ) -> None:
        node_editor.add_node(
            node_data(node)
        )

    def add_edge(
        edge: PipelineEdge,
    ) -> None:
        async def deferred_add():
            # Give VueFlow time to measure
            # the newly-added custom node.
            await asyncio.sleep(0.05)

            node_editor.add_edge(
                edge_data(edge)
            )

            node_editor.fit_view()

        asyncio.create_task(
            deferred_add()
        )

    ctrl.pipeline_add_node = add_node
    ctrl.pipeline_add_edge = add_edge

    for node in pipeline.nodes.values():
        node_editor.add_node(
            node_data(node)
        )

    def initialize_edges():
        for edge in pipeline.edges:
            node_editor.add_edge(
                edge_data(edge)
            )

        node_editor.fit_view()

    ctrl.on_client_connected.add(
        initialize_edges
    )

    return node_editor
