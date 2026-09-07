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
    normalize_property_value,
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
        self._modification_versions: dict[str, int] = {}
        self._property_descriptors: dict[str, dict[str, object]] = {}

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
        self._modification_versions.clear()
        self._property_descriptors.clear()
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
        self._modification_versions[node_id] = 1

        descriptors = inspect_properties(processor)
        self._property_descriptors[node_id] = {
            descriptor.name: descriptor for descriptor in descriptors
        }
        properties = {
            descriptor.name: self._property_state(descriptor, descriptor.value)
            for descriptor in descriptors
        }

        node = {
            "id": node_id,
            "name": (name or processor.GetClassName()),
            "class_name": (processor.GetClassName()),
            "input_port_count": (processor.GetNumberOfInputPorts()),
            "output_port_count": (processor.GetNumberOfOutputPorts()),
            "properties": properties,
            "input_arrays": self._inspect_input_array_state(processor),
            "execution_state": "modified",
        }

        pipeline_state = dict(self.state.pipeline)

        nodes = dict(pipeline_state["nodes"])

        nodes[node_id] = node

        pipeline_state["nodes"] = nodes

        self.state.pipeline = pipeline_state

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
        self._modification_versions.pop(node_id, None)
        self._property_descriptors.pop(node_id, None)

        for target_node_id in affected_targets:
            if target_node_id in nodes:
                self.mark_modified(target_node_id)
                self.bind_inputs(target_node_id)
                self.refresh_runtime_metadata(target_node_id)

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

        self.mark_modified(target_node_id)
        self.bind_inputs(target_node_id)

        self.refresh_runtime_metadata(target_node_id)

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

        self.mark_modified(edge.target_node_id)
        self.bind_inputs(edge.target_node_id)
        self.refresh_runtime_metadata(edge.target_node_id)

    def incoming_edges(
        self,
        node_id: str,
    ) -> list[PipelineEdge]:
        return [edge for edge in self.edges if edge.target_node_id == node_id]

    def outgoing_edges(
        self,
        node_id: str,
    ) -> list[PipelineEdge]:
        return [edge for edge in self.edges if edge.source_node_id == node_id]

    def bind_inputs(
        self,
        target_node_id: str,
    ) -> None:
        """Materialize model edges as concrete VTK data-object inputs.

        This deliberately performs *wiring only*. It never calls ``Update()`` or
        ``UpdateInformation()``. Connected filters can therefore inspect their
        current (possibly stale) input data for array metadata, bounds, and
        similar UI helpers without handing execution back to VTK's executive.
        """

        target = self.processor(target_node_id)

        for port in range(target.GetNumberOfInputPorts()):
            target.RemoveAllInputConnections(port)

        by_port: dict[int, list[vtk.vtkDataObject]] = {}
        for edge in self.incoming_edges(target_node_id):
            source = self.processor(edge.source_node_id)
            data = source.GetOutputDataObject(edge.source_port)
            if data is not None:
                by_port.setdefault(edge.target_port, []).append(data)

        for port, inputs in by_port.items():
            if not inputs:
                continue
            target.SetInputDataObject(port, inputs[0])
            for data in inputs[1:]:
                target.AddInputDataObject(port, data)

        target.Modified()

    def bind_downstream_inputs(
        self,
        source_node_id: str,
    ) -> None:
        """Rebind targets to the source node's latest output objects."""

        target_ids = {
            edge.target_node_id for edge in self.outgoing_edges(source_node_id)
        }
        for target_node_id in target_ids:
            self.bind_inputs(target_node_id)
            self.refresh_runtime_metadata(target_node_id)

    # -------------------------------------------------------------------------
    # Properties / input arrays
    # -------------------------------------------------------------------------

    def _property_state(self, descriptor, value) -> dict:
        return {
            "name": descriptor.name,
            "label": descriptor.label,
            "kind": descriptor.kind,
            "value": value,
            "size": descriptor.size,
        }

    def _inspect_input_array_state(
        self,
        processor: vtk.vtkAlgorithm,
        existing: dict | None = None,
    ) -> dict:
        """Read input-derived metadata while preserving model-owned selections."""

        existing = existing or {}
        result = {}
        for descriptor in inspect_input_arrays(processor):
            key = str(descriptor.index)
            previous = existing.get(key, {})
            result[key] = {
                "index": descriptor.index,
                "label": descriptor.label,
                "port": descriptor.port,
                "connection": descriptor.connection,
                "value": previous.get("value", descriptor.value),
                "items": descriptor.items,
            }
        return result

    def refresh_runtime_metadata(self, node_id: str) -> None:
        """Refresh only metadata that is genuinely derived from live VTK data.

        Processor property values are intentionally *not* read back here. vtkweb's
        model owns the user-requested property values; VTK is a consumer of that
        configuration. Runtime refresh is limited to information such as available
        input arrays and port counts that the model cannot know on its own.
        """

        processor = self.processor(node_id)
        pipeline_state = dict(self.state.pipeline)
        nodes = dict(pipeline_state["nodes"])
        node = dict(nodes[node_id])
        node.update(
            {
                "input_port_count": processor.GetNumberOfInputPorts(),
                "output_port_count": processor.GetNumberOfOutputPorts(),
                "input_arrays": self._inspect_input_array_state(
                    processor, node.get("input_arrays", {})
                ),
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
        descriptor = self._property_descriptors.get(node_id, {}).get(name)
        if descriptor is None:
            if value is None:
                return
            processor = self.processor(node_id)
            raise KeyError(
                f"Property {name!r} is not editable on {processor.GetClassName()}"
            )

        value = normalize_property_value(descriptor, value)

        # Model first: preserve the user's requested value even if a backend
        # clamps or otherwise interprets it differently. Deliberately do not
        # read other VTK getters back after the setter: some hand-written VTK
        # setters may affect coupled properties, but reflecting those side
        # effects would make backend behavior overwrite application intent. If
        # such coupling matters for a specific property, model it explicitly.
        pipeline_state = dict(self.state.pipeline)
        nodes = dict(pipeline_state["nodes"])
        node = dict(nodes[node_id])
        properties = dict(node.get("properties", {}))
        property_state = dict(properties[name])
        property_state["value"] = value
        if property_state.get("kind") == "scalar_list":
            property_state["size"] = len(value)
        properties[name] = property_state
        node["properties"] = properties
        nodes[node_id] = node
        pipeline_state["nodes"] = nodes
        self.state.pipeline = pipeline_state

        self.mark_modified(node_id)
        set_property(self.processor(node_id), descriptor, value)

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

        pipeline_state = dict(self.state.pipeline)
        nodes = dict(pipeline_state["nodes"])
        node = dict(nodes[node_id])
        input_arrays = dict(node.get("input_arrays", {}))
        key = str(int(index))
        input_state = dict(input_arrays.get(key, {}))
        input_state.update(
            {
                "index": descriptor.index,
                "label": descriptor.label,
                "port": descriptor.port,
                "connection": descriptor.connection,
                "value": value,
                "items": descriptor.items,
            }
        )
        input_arrays[key] = input_state
        node["input_arrays"] = input_arrays
        nodes[node_id] = node
        pipeline_state["nodes"] = nodes
        self.state.pipeline = pipeline_state

        self.mark_modified(node_id)
        set_input_array(processor, descriptor, value)

    # -------------------------------------------------------------------------
    # Execution state / graph traversal
    # -------------------------------------------------------------------------

    def execution_state(self, node_id: str) -> str:
        return self.node_state(node_id).get("execution_state", "modified")

    def modification_version(self, node_id: str) -> int:
        return self._modification_versions.get(node_id, 0)

    def set_execution_state(self, node_id: str, execution_state: str) -> None:
        pipeline_state = dict(self.state.pipeline)
        nodes = dict(pipeline_state["nodes"])
        node = dict(nodes[node_id])
        node["execution_state"] = execution_state
        nodes[node_id] = node
        pipeline_state["nodes"] = nodes
        self.state.pipeline = pipeline_state

    def set_execution_states(self, node_ids, execution_state: str) -> None:
        node_ids = list(node_ids)
        if not node_ids:
            return
        pipeline_state = dict(self.state.pipeline)
        nodes = dict(pipeline_state["nodes"])
        for node_id in node_ids:
            if node_id not in nodes:
                continue
            node = dict(nodes[node_id])
            node["execution_state"] = execution_state
            nodes[node_id] = node
        pipeline_state["nodes"] = nodes
        self.state.pipeline = pipeline_state

    def mark_modified(self, node_id: str, *, include_downstream: bool = True) -> None:
        affected = (
            self.downstream_subgraph(node_id) if include_downstream else {node_id}
        )
        pipeline_state = dict(self.state.pipeline)
        nodes = dict(pipeline_state["nodes"])
        for affected_id in affected:
            if affected_id not in nodes:
                continue
            self._modification_versions[affected_id] = (
                self.modification_version(affected_id) + 1
            )
            node = dict(nodes[affected_id])
            # Preserve running so the UI keeps showing execution; the version
            # makes the scheduler return it to modified after Update().
            if node.get("execution_state") != "running":
                node["execution_state"] = "modified"
            nodes[affected_id] = node
        pipeline_state["nodes"] = nodes
        self.state.pipeline = pipeline_state

    def modified_node_ids(self) -> list[str]:
        return [
            node_id
            for node_id in self.nodes
            if self.execution_state(node_id) == "modified"
        ]

    def downstream_subgraph(self, node_id: str) -> set[str]:
        result = {node_id}
        stack = [node_id]
        edges = self.edges
        while stack:
            current = stack.pop()
            for edge in edges:
                if edge.source_node_id == current and edge.target_node_id not in result:
                    result.add(edge.target_node_id)
                    stack.append(edge.target_node_id)
        return result

    def topological_order(self) -> list[str]:
        node_ids = list(self.state.pipeline["nodes"])
        indegree = {node_id: 0 for node_id in node_ids}
        outgoing = {node_id: [] for node_id in node_ids}
        for edge in self.edges:
            if (
                edge.source_node_id not in indegree
                or edge.target_node_id not in indegree
            ):
                continue
            indegree[edge.target_node_id] += 1
            outgoing[edge.source_node_id].append(edge.target_node_id)

        queue = [node_id for node_id in node_ids if indegree[node_id] == 0]
        order = []
        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            for target in outgoing[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)

        if len(order) != len(node_ids):
            raise RuntimeError("Pipeline graph contains a cycle")
        return order
