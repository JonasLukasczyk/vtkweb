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
    def processor(self) -> vtk.vtkAlgorithm:
        return self._graph.processor(self.id)

    @property
    def name(self) -> str:
        return self._graph.node_state(self.id)["name"]

    @property
    def class_name(self) -> str:
        return self._graph.node_state(self.id)["class_name"]


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
            source_node_id=value["source_node_id"],
            target_node_id=value["target_node_id"],
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

    The only persistent Python-only data kept here are live VTK processor
    instances. Everything describing the application-level pipeline is stored
    in ``state.pipeline`` so Vue and other UI components can react to the same
    source of truth.
    """

    def __init__(
        self,
        state,
    ) -> None:
        self.state = state
        self._processors: dict[
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
            for node_id in self.state.pipeline["nodes"]
        }

    @property
    def edges(
        self,
    ) -> list[PipelineEdge]:
        return [
            PipelineEdge.from_state(value) for value in self.state.pipeline["edges"]
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

        if node_id not in self.state.pipeline["nodes"]:
            return None

        return PipelineNode(
            self,
            node_id,
        )

    def node_state(
        self,
        node_id: str,
    ) -> dict:
        return self.state.pipeline["nodes"][node_id]

    def processor(
        self,
        node_id: str,
    ) -> vtk.vtkAlgorithm:
        return self._processors[node_id]

    def clear(
        self,
    ) -> None:
        """Clear pipeline state without inspecting or executing processors."""

        self._processors.clear()
        self.state.pipeline = {
            "nodes": {},
            "edges": [],
        }
        self.state.active_node_id = None

    # -------------------------------------------------------------------------
    # Nodes
    # -------------------------------------------------------------------------

    def add_node(
        self,
        processor: vtk.vtkAlgorithm,
        *,
        name: str | None = None,
        node_id: str | None = None,
    ) -> PipelineNode:
        node_id = node_id or uuid4().hex

        if node_id in self.state.pipeline["nodes"] or node_id in self._processors:
            raise ValueError(f"Node ID already exists: {node_id}")

        self._processors[node_id] = processor

        node = {
            "id": node_id,
            "name": (name or processor.GetClassName()),
            "class_name": (processor.GetClassName()),
            "input_port_count": (processor.GetNumberOfInputPorts()),
            "output_port_count": (processor.GetNumberOfOutputPorts()),
            "properties": {},
            "input_arrays": {},
        }

        pipeline_state = dict(self.state.pipeline)

        nodes = dict(pipeline_state["nodes"])

        nodes[node_id] = node

        pipeline_state["nodes"] = nodes

        self.state.pipeline = pipeline_state

        # IMPORTANT:
        #
        # Do not call sync_node_from_runtime() here.
        #
        # A newly-created filter may have required input ports that have not
        # been connected yet. sync_node_from_runtime() inspects input arrays,
        # which can cause VTK to call UpdateInformation() on an incomplete
        # pipeline and produce errors such as:
        #
        #   Input port 0 ... has 0 connections but is not optional.
        #
        # Runtime synchronization happens after the node is connected.

        if self.active_node_id is None:
            self.state.active_node_id = node_id

        return PipelineNode(
            self,
            node_id,
        )

    def remove_node(
        self,
        node_id: str,
    ) -> None:
        if node_id not in self.state.pipeline["nodes"]:
            return

        affected_targets = {
            edge.target_node_id
            for edge in self.edges
            if (edge.source_node_id == node_id or edge.target_node_id == node_id)
        }

        pipeline_state = dict(self.state.pipeline)

        nodes = dict(pipeline_state["nodes"])

        nodes.pop(
            node_id,
            None,
        )

        pipeline_state["nodes"] = nodes

        pipeline_state["edges"] = [
            edge
            for edge in pipeline_state["edges"]
            if (edge["source_node_id"] != node_id and edge["target_node_id"] != node_id)
        ]

        self.state.pipeline = pipeline_state

        self._processors.pop(
            node_id,
            None,
        )

        for target_node_id in affected_targets:
            if target_node_id in nodes:
                self._sync_inputs(target_node_id)
                self.sync_node_from_runtime(target_node_id)

        if self.active_node_id == node_id:
            self.state.active_node_id = next(
                iter(nodes),
                None,
            )

    def set_active_node(
        self,
        node_id: str | None,
    ) -> None:
        if node_id is not None and node_id not in self.state.pipeline["nodes"]:
            raise KeyError(node_id)

        self.state.active_node_id = node_id

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
        sync: bool = True,
    ) -> PipelineEdge:
        edge = PipelineEdge(
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            source_port=source_port,
            target_port=target_port,
        )

        pipeline_state = dict(self.state.pipeline)

        edges = list(pipeline_state["edges"])

        edges.append(edge.to_state())

        pipeline_state["edges"] = edges

        self.state.pipeline = pipeline_state

        self._sync_inputs(target_node_id)

        if sync:
            self.sync_node_from_runtime(target_node_id)

        return edge

    def disconnect(
        self,
        edge: PipelineEdge,
    ) -> None:
        edge_state = edge.to_state()

        pipeline_state = dict(self.state.pipeline)

        edges = list(pipeline_state["edges"])

        edges.remove(edge_state)

        pipeline_state["edges"] = edges

        self.state.pipeline = pipeline_state

        self._sync_inputs(edge.target_node_id)

        self.sync_node_from_runtime(edge.target_node_id)

    def incoming_edges(
        self,
        node_id: str,
    ) -> list[PipelineEdge]:
        return [edge for edge in self.edges if edge.target_node_id == node_id]

    # -------------------------------------------------------------------------
    # Properties / input arrays
    # -------------------------------------------------------------------------

    def sync_node_from_runtime(
        self,
        node_id: str,
    ) -> None:
        """Pull processor metadata/properties into authoritative UI state.

        Normal vtkweb mutations should go through the methods below, which
        automatically keep state synchronized. This explicit method also gives
        callers a supported escape hatch after intentionally mutating a raw VTK
        processor directly.
        """

        processor = self.processor(node_id)

        properties = {
            descriptor.name: {
                "name": descriptor.name,
                "label": descriptor.label,
                "kind": descriptor.kind,
                "value": descriptor.value,
                "size": descriptor.size,
            }
            for descriptor in inspect_properties(processor)
        }

        input_arrays = {
            str(descriptor.index): {
                "index": descriptor.index,
                "label": descriptor.label,
                "port": descriptor.port,
                "connection": (descriptor.connection),
                "value": descriptor.value,
                "items": descriptor.items,
            }
            for descriptor in inspect_input_arrays(processor)
        }

        pipeline_state = dict(self.state.pipeline)

        nodes = dict(pipeline_state["nodes"])

        node = dict(nodes[node_id])

        node.update(
            {
                "name": node.get(
                    "name",
                    processor.GetClassName(),
                ),
                "class_name": (processor.GetClassName()),
                "input_port_count": (processor.GetNumberOfInputPorts()),
                "output_port_count": (processor.GetNumberOfOutputPorts()),
                "properties": properties,
                "input_arrays": input_arrays,
            }
        )

        nodes[node_id] = node

        pipeline_state["nodes"] = nodes

        self.state.pipeline = pipeline_state

    def set_property(
        self,
        node_id: str,
        name: str,
        value,
    ) -> None:
        processor = self.processor(node_id)

        descriptor = next(
            descriptor
            for descriptor in inspect_properties(processor)
            if descriptor.name == name
        )

        set_property(
            processor,
            descriptor,
            value,
        )

        processor.Update()

        self.sync_node_from_runtime(node_id)

    def set_vector_component(
        self,
        node_id: str,
        name: str,
        index: int,
        value,
    ) -> None:
        property_state = self.node_state(node_id)["properties"][name]

        values = list(property_state["value"])

        values[int(index)] = float(value)

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

        property_state = self.node_state(node_id)["properties"][name]

        values = list(property_state["value"])

        values[int(index)] = float(value)

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
        property_state = self.node_state(node_id)["properties"][name]

        values = list(property_state["value"])

        values.append(values[-1] if values else 0.0)

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
        property_state = self.node_state(node_id)["properties"][name]

        values = list(property_state["value"])

        index = int(index)

        if index < 0 or index >= len(values):
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

        processor = self.processor(node_id)

        descriptor = next(
            descriptor
            for descriptor in inspect_input_arrays(processor)
            if descriptor.index == int(index)
        )

        set_input_array(
            processor,
            descriptor,
            value,
        )

        processor.Update()

        self.sync_node_from_runtime(node_id)

    # -------------------------------------------------------------------------
    # Runtime VTK connectivity
    # -------------------------------------------------------------------------

    def _sync_inputs(
        self,
        target_node_id: str,
    ) -> None:
        target = self.processor(target_node_id)

        for port in range(target.GetNumberOfInputPorts()):
            target.RemoveAllInputConnections(port)

        for edge in self.incoming_edges(target_node_id):
            source = self.processor(edge.source_node_id)

            target.AddInputConnection(
                edge.target_port,
                source.GetOutputPort(edge.source_port),
            )

        target.Modified()
