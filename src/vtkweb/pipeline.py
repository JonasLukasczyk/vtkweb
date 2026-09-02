from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

import vtk


@dataclass
class PipelineNode:
    algorithm: vtk.vtkAlgorithm
    name: str | None = None
    visible: bool = False
    id: str = field(
        default_factory=lambda: uuid4().hex
    )

    def __post_init__(self) -> None:
        self.name = (
            self.name
            or self.algorithm.GetClassName()
        )


@dataclass(frozen=True)
class PipelineEdge:
    source_node_id: str
    target_node_id: str
    source_port: int = 0
    target_port: int = 0


class PipelineGraph:
    def __init__(self) -> None:
        self.nodes: dict[
            str,
            PipelineNode,
        ] = {}

        self.edges: list[
            PipelineEdge
        ] = []

        self.active_node_id: str | None = None

    @property
    def active_node(
        self,
    ) -> PipelineNode | None:
        if self.active_node_id is None:
            return None

        return self.nodes.get(
            self.active_node_id
        )

    def add_node(
        self,
        algorithm: vtk.vtkAlgorithm,
        *,
        name: str | None = None,
        visible: bool = False,
    ) -> PipelineNode:
        node = PipelineNode(
            algorithm=algorithm,
            name=name,
            visible=visible,
        )

        self.nodes[node.id] = node

        if self.active_node_id is None:
            self.active_node_id = node.id

        return node

    def remove_node(
        self,
        node_id: str,
    ) -> None:
        del self.nodes[node_id]

        affected_targets = {
            edge.target_node_id
            for edge in self.edges
            if edge.source_node_id == node_id
            or edge.target_node_id == node_id
        }

        self.edges = [
            edge
            for edge in self.edges
            if edge.source_node_id != node_id
            and edge.target_node_id != node_id
        ]

        for target_node_id in affected_targets:
            if target_node_id in self.nodes:
                self._sync_inputs(
                    target_node_id
                )

        if self.active_node_id == node_id:
            self.active_node_id = next(
                iter(self.nodes),
                None,
            )

    def set_active_node(
        self,
        node_id: str | None,
    ) -> None:
        self.active_node_id = node_id

    def connect(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        source_port: int = 0,
        target_port: int = 0,
    ) -> PipelineEdge:
        edge = PipelineEdge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            source_port=source_port,
            target_port=target_port,
        )

        self.edges.append(edge)

        self._sync_inputs(
            target_node_id
        )

        return edge

    def disconnect(
        self,
        edge: PipelineEdge,
    ) -> None:
        self.edges.remove(edge)

        self._sync_inputs(
            edge.target_node_id
        )

    def incoming_edges(
        self,
        node_id: str,
    ) -> list[PipelineEdge]:
        return [
            edge
            for edge in self.edges
            if edge.target_node_id == node_id
        ]

    def _sync_inputs(
        self,
        target_node_id: str,
    ) -> None:
        target = self.nodes[
            target_node_id
        ].algorithm

        for port in range(
            target.GetNumberOfInputPorts()
        ):
            target.RemoveAllInputConnections(
                port
            )

        for edge in self.incoming_edges(
            target_node_id
        ):
            source = self.nodes[
                edge.source_node_id
            ].algorithm

            target.AddInputConnection(
                edge.target_port,
                source.GetOutputPort(
                    edge.source_port
                ),
            )

        target.Modified()
