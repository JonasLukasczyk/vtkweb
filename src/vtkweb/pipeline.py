from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import vtk

from vtkweb.input_arrays import (
    inspect_input_arrays,
    set_input_array,
)
from vtkweb.properties import (
    inspect_properties,
    set_property,
)


class PipelineNode:
    """Lightweight runtime view onto one node stored in trame state."""

    def __init__(
        self,
        graph: "PipelineGraph",
        node_id: str,
    ) -> None:
        self._graph = graph
        self.id = node_id

    @property
    def algorithm(self) -> vtk.vtkAlgorithm:
        return self._graph.algorithm(
            self.id
        )

    @property
    def name(self) -> str:
        return self._graph.node_state(
            self.id
        )["name"]

    @property
    def class_name(self) -> str:
        return self._graph.node_state(
            self.id
        )["class_name"]


@dataclass(frozen=True)
class PipelineEdge:
    source_node_id: str
    target_node_id: str
    source_port: int = 0
    target_port: int = 0

    def to_state(self) -> dict:
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "source_port": self.source_port,
            "target_port": self.target_port,
        }

    @classmethod
    def from_state(
        cls,
        value: dict,
    ) -> "PipelineEdge":
        return cls(
            source_node_id=value[
                "source_node_id"
            ],
            target_node_id=value[
                "target_node_id"
            ],
            source_port=int(
                value.get(
                    "source_port",
                    0,
                )
            ),
            target_port=int(
                value.get(
                    "target_port",
                    0,
                )
            ),
        )


class PipelineGraph:
    """Pipeline service whose serializable model lives in trame state.

    The only persistent Python-only data kept here are live VTK algorithm
    instances. Everything describing the application-level pipeline is stored
    in ``state.pipeline`` so Vue and other UI components can react to the same
    source of truth.
    """

    def __init__(
        self,
        state,
    ) -> None:
        self.state = state
        self._algorithms: dict[
            str,
            vtk.vtkAlgorithm,
        ] = {}

        self.state.pipeline = {
            "nodes": {},
            "edges": [],
        }
        self.state.active_node_id = None

    # -------------------------------------------------------------------------
    # State access
    # -------------------------------------------------------------------------

    @property
    def nodes(
        self,
    ) -> dict[str, PipelineNode]:
        return {
            node_id: PipelineNode(
                self,
                node_id,
            )
            for node_id in self.state.pipeline[
                "nodes"
            ]
        }

    @property
    def edges(
        self,
    ) -> list[PipelineEdge]:
        return [
            PipelineEdge.from_state(
                value
            )
            for value in self.state.pipeline[
                "edges"
            ]
        ]

    @property
    def active_node_id(
        self,
    ) -> str | None:
        return self.state.active_node_id

    @property
    def active_node(
        self,
    ) -> PipelineNode | None:
        node_id = self.active_node_id

        if node_id is None:
            return None

        if node_id not in self.state.pipeline[
            "nodes"
        ]:
            return None

        return PipelineNode(
            self,
            node_id,
        )

    def node_state(
        self,
        node_id: str,
    ) -> dict:
        return self.state.pipeline[
            "nodes"
        ][node_id]

    def algorithm(
        self,
        node_id: str,
    ) -> vtk.vtkAlgorithm:
        return self._algorithms[
            node_id
        ]

    # -------------------------------------------------------------------------
    # Nodes
    # -------------------------------------------------------------------------

    def add_node(
        self,
        algorithm: vtk.vtkAlgorithm,
        *,
        name: str | None = None,
    ) -> PipelineNode:
        node_id = uuid4().hex

        self._algorithms[
            node_id
        ] = algorithm

        node = {
            "id": node_id,
            "name": (
                name
                or algorithm.GetClassName()
            ),
            "class_name": (
                algorithm.GetClassName()
            ),
            "input_port_count": (
                algorithm.GetNumberOfInputPorts()
            ),
            "output_port_count": (
                algorithm.GetNumberOfOutputPorts()
            ),
            "properties": {},
            "input_arrays": {},
        }

        pipeline_state = dict(
            self.state.pipeline
        )
        nodes = dict(
            pipeline_state["nodes"]
        )
        nodes[node_id] = node
        pipeline_state["nodes"] = nodes

        self.state.pipeline = (
            pipeline_state
        )

        self.sync_node_from_runtime(
            node_id
        )

        if self.active_node_id is None:
            self.state.active_node_id = (
                node_id
            )

        return PipelineNode(
            self,
            node_id,
        )

    def remove_node(
        self,
        node_id: str,
    ) -> None:
        if node_id not in self.state.pipeline[
            "nodes"
        ]:
            return

        affected_targets = {
            edge.target_node_id
            for edge in self.edges
            if (
                edge.source_node_id
                == node_id
                or edge.target_node_id
                == node_id
            )
        }

        pipeline_state = dict(
            self.state.pipeline
        )

        nodes = dict(
            pipeline_state["nodes"]
        )
        nodes.pop(
            node_id,
            None,
        )

        pipeline_state["nodes"] = (
            nodes
        )

        pipeline_state["edges"] = [
            edge
            for edge in pipeline_state[
                "edges"
            ]
            if (
                edge["source_node_id"]
                != node_id
                and edge[
                    "target_node_id"
                ]
                != node_id
            )
        ]

        self.state.pipeline = (
            pipeline_state
        )

        self._algorithms.pop(
            node_id,
            None,
        )

        for target_node_id in (
            affected_targets
        ):
            if target_node_id in nodes:
                self._sync_inputs(
                    target_node_id
                )
                self.sync_node_from_runtime(
                    target_node_id
                )

        if self.active_node_id == node_id:
            self.state.active_node_id = next(
                iter(nodes),
                None,
            )

    def set_active_node(
        self,
        node_id: str | None,
    ) -> None:
        if (
            node_id is not None
            and node_id
            not in self.state.pipeline[
                "nodes"
            ]
        ):
            raise KeyError(
                node_id
            )

        self.state.active_node_id = (
            node_id
        )

    # -------------------------------------------------------------------------
    # Edges
    # -------------------------------------------------------------------------

    def connect(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        source_port: int = 0,
        target_port: int = 0,
    ) -> PipelineEdge:
        edge = PipelineEdge(
            source_node_id=(
                source_node_id
            ),
            target_node_id=(
                target_node_id
            ),
            source_port=source_port,
            target_port=target_port,
        )

        pipeline_state = dict(
            self.state.pipeline
        )
        edges = list(
            pipeline_state["edges"]
        )
        edges.append(
            edge.to_state()
        )
        pipeline_state["edges"] = (
            edges
        )
        self.state.pipeline = (
            pipeline_state
        )

        self._sync_inputs(
            target_node_id
        )
        self.sync_node_from_runtime(
            target_node_id
        )

        return edge

    def disconnect(
        self,
        edge: PipelineEdge,
    ) -> None:
        edge_state = edge.to_state()

        pipeline_state = dict(
            self.state.pipeline
        )
        edges = list(
            pipeline_state["edges"]
        )
        edges.remove(
            edge_state
        )
        pipeline_state["edges"] = (
            edges
        )
        self.state.pipeline = (
            pipeline_state
        )

        self._sync_inputs(
            edge.target_node_id
        )
        self.sync_node_from_runtime(
            edge.target_node_id
        )

    def incoming_edges(
        self,
        node_id: str,
    ) -> list[PipelineEdge]:
        return [
            edge
            for edge in self.edges
            if edge.target_node_id
            == node_id
        ]

    # -------------------------------------------------------------------------
    # Properties / input arrays
    # -------------------------------------------------------------------------

    def sync_node_from_runtime(
        self,
        node_id: str,
    ) -> None:
        """Pull algorithm metadata/properties into authoritative UI state.

        Normal vtkweb mutations should go through the methods below, which
        automatically keep state synchronized. This explicit method also gives
        callers a supported escape hatch after intentionally mutating a raw VTK
        algorithm directly.
        """

        algorithm = self.algorithm(
            node_id
        )

        properties = {
            descriptor.name: {
                "name": descriptor.name,
                "label": descriptor.label,
                "kind": descriptor.kind,
                "value": descriptor.value,
                "size": descriptor.size,
            }
            for descriptor in inspect_properties(
                algorithm
            )
        }

        input_arrays = {
            str(descriptor.index): {
                "index": descriptor.index,
                "label": descriptor.label,
                "port": descriptor.port,
                "connection": (
                    descriptor.connection
                ),
                "value": descriptor.value,
                "items": descriptor.items,
            }
            for descriptor in inspect_input_arrays(
                algorithm
            )
        }

        pipeline_state = dict(
            self.state.pipeline
        )
        nodes = dict(
            pipeline_state["nodes"]
        )
        node = dict(
            nodes[node_id]
        )

        node.update(
            {
                "name": node.get(
                    "name",
                    algorithm.GetClassName(),
                ),
                "class_name": (
                    algorithm.GetClassName()
                ),
                "input_port_count": (
                    algorithm.GetNumberOfInputPorts()
                ),
                "output_port_count": (
                    algorithm.GetNumberOfOutputPorts()
                ),
                "properties": properties,
                "input_arrays": input_arrays,
            }
        )

        nodes[node_id] = node
        pipeline_state["nodes"] = nodes
        self.state.pipeline = (
            pipeline_state
        )

    def set_property(
        self,
        node_id: str,
        name: str,
        value,
    ) -> None:
        algorithm = self.algorithm(
            node_id
        )

        descriptor = next(
            descriptor
            for descriptor in inspect_properties(
                algorithm
            )
            if descriptor.name == name
        )

        set_property(
            algorithm,
            descriptor,
            value,
        )

        algorithm.Update()
        self.sync_node_from_runtime(
            node_id
        )

    def set_vector_component(
        self,
        node_id: str,
        name: str,
        index: int,
        value,
    ) -> None:
        property_state = (
            self.node_state(node_id)[
                "properties"
            ][name]
        )
        values = list(
            property_state["value"]
        )
        values[int(index)] = float(
            value
        )
        self.set_property(
            node_id,
            name,
            values,
        )

    def set_list_value(
        self,
        node_id: str,
        name: str,
        index: int,
        value,
    ) -> None:
        if value in ("", None):
            return

        property_state = (
            self.node_state(node_id)[
                "properties"
            ][name]
        )
        values = list(
            property_state["value"]
        )
        values[int(index)] = float(
            value
        )
        self.set_property(
            node_id,
            name,
            values,
        )

    def add_list_value(
        self,
        node_id: str,
        name: str,
    ) -> None:
        property_state = (
            self.node_state(node_id)[
                "properties"
            ][name]
        )
        values = list(
            property_state["value"]
        )
        values.append(
            values[-1]
            if values
            else 0.0
        )
        self.set_property(
            node_id,
            name,
            values,
        )

    def remove_list_value(
        self,
        node_id: str,
        name: str,
        index: int,
    ) -> None:
        property_state = (
            self.node_state(node_id)[
                "properties"
            ][name]
        )
        values = list(
            property_state["value"]
        )
        index = int(index)

        if (
            index < 0
            or index >= len(values)
        ):
            return

        del values[index]
        self.set_property(
            node_id,
            name,
            values,
        )

    def set_input_array(
        self,
        node_id: str,
        index: int,
        value,
    ) -> None:
        if not value:
            return

        algorithm = self.algorithm(
            node_id
        )

        descriptor = next(
            descriptor
            for descriptor in inspect_input_arrays(
                algorithm
            )
            if descriptor.index == int(index)
        )

        set_input_array(
            algorithm,
            descriptor,
            value,
        )

        algorithm.Update()
        self.sync_node_from_runtime(
            node_id
        )

    # -------------------------------------------------------------------------
    # Runtime VTK connectivity
    # -------------------------------------------------------------------------

    def _sync_inputs(
        self,
        target_node_id: str,
    ) -> None:
        target = self.algorithm(
            target_node_id
        )

        for port in range(
            target.GetNumberOfInputPorts()
        ):
            target.RemoveAllInputConnections(
                port
            )

        for edge in self.incoming_edges(
            target_node_id
        ):
            source = self.algorithm(
                edge.source_node_id
            )

            target.AddInputConnection(
                edge.target_port,
                source.GetOutputPort(
                    edge.source_port
                ),
            )

        target.Modified()
