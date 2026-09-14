from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
from copy import deepcopy
from uuid import NAMESPACE_URL, uuid4, uuid5

from vtkweb.pipeline import PipelineGraph
from vtkweb.distributed import context as distributed
from vtkweb.rendering.base import (
    DEFAULT_VIEW_PROPERTIES,
    REPRESENTATION_KINDS,
    VIEW_PROPERTY_NAMES,
    view_property_state,
    RenderView,
    FrameRenderingBackend,
    RenderingBackend,
    Representation,
)
from vtkweb.rendering.frame_scheduler import FrameRenderManager
from vtkweb.rendering.vtk_backend import (
    VTKRenderingBackend,
)
from vtkweb.transfer_functions import TransferFunctionManager


DEFAULT_REPRESENTATION_PROPERTIES = {
    "color_by": None,
    "color": "#ffffff",
    "line_width": 0.01,
    "tube_sides": 3,
    "interpolation": "linear",
    "blend_mode": "composite",
    "shade": True,
    "ambient": 0.1,
    "diffuse": 0.9,
    "specular": 0.2,
    "specular_power": 10.0,
    "global_illumination_reach": 0.0,
    "volumetric_scattering_blending": 0.0,
    "auto_adjust_sample_distances": True,
    "sample_distance": 1.0,
}

DEFAULT_VIEW_CAMERAS = {
    "vtk": {
        "position": [0.0, 0.0, 1.0],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
        "fov": 30.0,
        "parallel_projection": False,
        "parallel_scale": 1.0,
    },
    "mitsuba": {
        "position": [0.0, 0.0, 5.0],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
        "center_of_rotation": [0.0, 0.0, 0.0],
        "fov": 30.0,
    },
}


class RenderManager:
    """Rendering service backed by serializable trame state.

    ``state.views`` and ``state.representations`` are the authoritative
    application model. Backend objects such as vtkActor and vtkRenderWindow are
    runtime-only materializations of that state.
    """

    def __init__(
        self,
        state,
        pipeline: PipelineGraph,
        backend: RenderingBackend | None = None,
        frame_transport=None,
    ) -> None:
        self.state = state
        self.pipeline = pipeline
        self.frames = FrameRenderManager(frame_transport)
        self.transfer_functions = TransferFunctionManager(state, self)
        vtk_backend = backend or VTKRenderingBackend(self.transfer_functions.get)
        self._backends: dict[str, RenderingBackend] = {"vtk": vtk_backend}
        self._view_backend_ids: dict[str, str] = {}
        # Client viewport size is transient runtime state owned by the logical
        # view. Keep it here so a backend switch does not lose the existing
        # canvas dimensions when the DOM itself does not resize.
        self._render_sizes: dict[str, tuple[int, int]] = {}
        # Transient size revisions are tracked for every logical view, even on
        # workers where the backend is not materialized. Because resize
        # mutations are replicated in order, all ranks keep the same revision
        # and late-materialized workers join the current framebuffer epoch.
        self._render_size_revisions: dict[str, int] = {}

        self.state.views = {}
        self.state.representations = {}
        self.state.active_view_id = None

    # -------------------------------------------------------------------------
    # Views
    # -------------------------------------------------------------------------

    @property
    def views(
        self,
    ) -> tuple[RenderView, ...]:
        return tuple(
            self.get_view(view_id)
            for view_id, value in self.state.views.items()
            if value.get("type") in {"vtk", "mitsuba"}
        )

    @property
    def active_view_id(
        self,
    ) -> str | None:
        return self.state.active_view_id

    def get_view(self, view_id: str) -> RenderView:
        value = self.state.views[view_id]
        if value.get("type") not in {"vtk", "mitsuba"}:
            raise ValueError(f"View is not a render view: {view_id}")
        return RenderView(id=value["id"], name=value["name"])

    def backend_view_id(self, view_id: str) -> str:
        self.get_view(view_id)
        return self._view_backend_ids[view_id]

    def add_view(
        self,
        name: str | None = None,
        *,
        view_id: str | None = None,
        view_type: str = "vtk",
    ) -> RenderView:
        if view_type not in {"vtk", "mitsuba"}:
            raise ValueError(f"Unknown render view type: {view_type}")
        if name is None:
            name = f"View {len(self.views) + 1}"

        view_id = view_id or uuid4().hex
        if view_id in self.state.views:
            raise ValueError(f"View ID already exists: {view_id}")

        property_values = {
            **DEFAULT_VIEW_PROPERTIES,
            "camera": deepcopy(DEFAULT_VIEW_CAMERAS[view_type]),
        }
        value = {
            "id": view_id,
            "type": view_type,
            "name": name,
            "properties": {
                property_name: view_property_state(
                    property_name, property_values[property_name]
                )
                for property_name in VIEW_PROPERTY_NAMES
            },
        }
        self.state.views = {**self.state.views, view_id: value}

        # Rank 0 always materializes browser-visible views. Worker ranks keep
        # only logical replicated state until Distributed Rendering is enabled.
        if self._should_materialize_view(view_id):
            self._materialize_view(view_id)

        self._notify_render()
        return self.get_view(view_id)

    def remove_view(
        self,
        view_id: str,
        *,
        preserve_render_size: bool = False,
    ) -> None:
        self.get_view(view_id)

        for representation in tuple(self.representations):
            if view_id in representation.view_ids:
                self.unassign_representation(representation.id, view_id, notify=False)

        if self._is_view_materialized(view_id):
            self._dematerialize_view(view_id)
        if not preserve_render_size:
            self._render_sizes.pop(view_id, None)
            self._render_size_revisions.pop(view_id, None)

        views = dict(self.state.views)
        del views[view_id]
        self.state.views = views

        if self.active_view_id == view_id:
            self.state.active_view_id = next((view.id for view in self.views), None)

        self._notify_render()

    def set_active_view(
        self,
        view_id: str,
    ) -> None:
        self.get_view(view_id)
        self.state.active_view_id = view_id

    def switch_view_type(self, view_id: str, view_type: str) -> None:
        """Replace a render backend in place while preserving logical state."""
        if view_type not in {"vtk", "mitsuba"}:
            raise ValueError(f"Unknown render view type: {view_type}")
        value = deepcopy(self.state.views[view_id])
        if value.get("type") == view_type:
            return

        was_materialized = self._is_view_materialized(view_id)
        if was_materialized:
            self._dematerialize_view(view_id)

        value["type"] = view_type
        # Keep the current camera when switching backends; it is renderer-agnostic.
        self.state.views = {**self.state.views, view_id: value}

        if self._should_materialize_view(view_id):
            self._materialize_view(view_id)
        self._notify_render()

    # -------------------------------------------------------------------------
    # Representations
    # -------------------------------------------------------------------------

    @property
    def representations(
        self,
    ) -> tuple[Representation, ...]:
        return tuple(
            self.get_representation(representation_id)
            for representation_id in self.state.representations
        )

    def get_representation(
        self,
        representation_id: str,
    ) -> Representation:
        value = self.state.representations[representation_id]
        properties = dict(DEFAULT_REPRESENTATION_PROPERTIES)
        properties.update(deepcopy(value.get("properties", {})))

        return Representation(
            id=value["id"],
            node_id=value["node_id"],
            output_port=int(value["output_port"]),
            kind=value["kind"],
            properties=properties,
            view_ids=set(value.get("view_ids", [])),
        )

    def get_representations(
        self,
        node_id: str,
        output_port: int | None = None,
    ) -> tuple[Representation, ...]:
        return tuple(
            representation
            for representation in self.representations
            if (
                representation.node_id == node_id
                and (output_port is None or representation.output_port == output_port)
            )
        )

    def add_representation(
        self,
        node_id: str,
        *,
        output_port: int = 0,
        kind: str = "surface",
        view_ids: Iterable[str] = (),
        camera_reset_mode: int = 0,
        representation_id: str | None = None,
        notify: bool = True,
    ) -> Representation:
        if kind not in REPRESENTATION_KINDS:
            raise ValueError(f"Unknown representation kind: {kind}")

        if camera_reset_mode not in (0, 1, 2):
            raise ValueError(
                f"Invalid camera_reset_mode: {camera_reset_mode}; expected 0, 1, or 2"
            )

        node = self.pipeline.nodes[node_id]
        output_count = node.processor.GetNumberOfOutputPorts()

        if output_port < 0 or output_port >= output_count:
            raise ValueError(
                f"{node.name} has "
                f"{output_count} output port(s); "
                f"port {output_port} is invalid"
            )

        representation_id = representation_id or uuid4().hex
        view_ids = tuple(view_ids)
        if representation_id in self.state.representations:
            raise ValueError(f"Representation ID already exists: {representation_id}")

        properties = dict(DEFAULT_REPRESENTATION_PROPERTIES)
        value = {
            "id": representation_id,
            "node_id": node_id,
            "output_port": int(output_port),
            "kind": kind,
            "properties": properties,
            "view_ids": [],
        }

        if kind == "volume":
            arrays = self.get_arrays(node_id, int(output_port))
            association = "point" if arrays["point"] else "cell"
            names = arrays[association]
            if names:
                properties["color_by"] = [names[0], association]
                self.transfer_functions.ensure(
                    names[0],
                    self.get_array_range(
                        node_id, int(output_port), names[0], association
                    ),
                )

        self.state.representations = {
            **self.state.representations,
            representation_id: value,
        }

        for view_id in view_ids:
            self.assign_representation(
                representation_id,
                view_id,
                notify=False,
            )

        if camera_reset_mode:
            for view_id in view_ids:
                representation_count = sum(
                    view_id in item.view_ids for item in self.representations
                )
                if camera_reset_mode == 2 or representation_count == 1:
                    self.reset_camera(view_id, notify=False)

        if notify and view_ids:
            self._notify_render()
        return self.get_representation(representation_id)

    def remove_representation(
        self,
        representation_id: str,
    ) -> None:
        representation = self.get_representation(representation_id)

        for view_id in tuple(representation.view_ids):
            self.unassign_representation(
                representation_id,
                view_id,
                notify=False,
            )

        representations = dict(self.state.representations)
        representations.pop(
            representation_id,
            None,
        )
        self.state.representations = representations
        self._notify_render()

    def remove_node(
        self,
        node_id: str,
    ) -> None:
        for representation in list(self.get_representations(node_id)):
            self.remove_representation(representation.id)

    def ensure_output_representations(
        self,
        node_id: str,
    ) -> tuple[str, ...]:
        """Create missing post-execution output representations as outlines.

        Nodes intentionally have no default representations while they are
        merely configured. After a successful execution, each output port that
        still has no representation receives exactly one outline representation.
        If the active view is a VTK view, the new representation is shown there.
        """

        node = self.pipeline.nodes[node_id]
        active_view_id = self.active_view_id
        view_ids: tuple[str, ...] = ()
        if active_view_id is not None:
            view = self.state.views.get(active_view_id)
            if view is not None and view.get("type") in {"vtk", "mitsuba"}:
                view_ids = (active_view_id,)

        created = []
        for output_port in range(node.processor.GetNumberOfOutputPorts()):
            if self.get_representations(node_id, output_port):
                continue
            representation_id = uuid5(
                NAMESPACE_URL, f"vtkweb:{node_id}:{output_port}:outline"
            ).hex
            representation = self.add_representation(
                node_id,
                output_port=output_port,
                kind="outline",
                view_ids=view_ids,
                representation_id=representation_id,
                notify=False,
            )
            created.append(representation.id)

        return tuple(created)

    def refresh_node(
        self,
        node_id: str,
    ) -> None:
        """Refresh every render representation backed by *node_id*."""

        for representation in tuple(self.get_representations(node_id)):
            self._update_representation(representation.id)

        self._notify_render()

    # -------------------------------------------------------------------------
    # View assignment / visibility
    # -------------------------------------------------------------------------

    def representation_in_view(
        self,
        representation_id: str,
        view_id: str,
    ) -> bool:
        return view_id in self.get_representation(representation_id).view_ids

    def assign_representation(
        self,
        representation_id: str,
        view_id: str,
        *,
        notify: bool = True,
    ) -> None:
        representation = self.get_representation(representation_id)

        if view_id in representation.view_ids:
            return

        self.get_view(view_id)
        if self._is_view_materialized(view_id) and self.pipeline.has_valid_output(
            representation.node_id
        ):
            node = self.pipeline.nodes[representation.node_id]
            self._backend_for_view(view_id).add_representation(
                representation, self._backend_view(view_id), node.processor
            )
            self.frames.ensure(view_id)

        value = dict(self.state.representations[representation_id])
        value["view_ids"] = [
            *value.get(
                "view_ids",
                [],
            ),
            view_id,
        ]
        self._set_representation_state(
            representation_id,
            value,
        )
        if notify:
            self._notify_render()

    def unassign_representation(
        self,
        representation_id: str,
        view_id: str,
        *,
        notify: bool = True,
    ) -> None:
        representation = self.get_representation(representation_id)

        if view_id not in representation.view_ids:
            return

        if self._is_view_materialized(view_id):
            self._backend_for_view(view_id).remove_representation(
                representation.id,
                self.backend_view_id(view_id),
            )

        value = dict(self.state.representations[representation_id])
        value["view_ids"] = [
            item
            for item in value.get(
                "view_ids",
                [],
            )
            if item != view_id
        ]
        self._set_representation_state(
            representation_id,
            value,
        )
        if notify:
            self._notify_render()

    # -------------------------------------------------------------------------
    # Representation properties
    # -------------------------------------------------------------------------

    def set_representation_kind(
        self,
        representation_id: str,
        kind: str,
    ) -> None:
        if kind not in REPRESENTATION_KINDS:
            raise ValueError(f"Unknown representation kind: {kind}")

        value = dict(self.state.representations[representation_id])
        value["kind"] = kind
        properties = dict(DEFAULT_REPRESENTATION_PROPERTIES)
        properties.update(value.get("properties", {}))
        if kind == "volume" and properties.get("color_by") is None:
            representation = self.get_representation(representation_id)
            arrays = self.get_arrays(representation.node_id, representation.output_port)
            association = "point" if arrays["point"] else "cell"
            names = arrays[association]
            if names:
                properties["color_by"] = [names[0], association]
                self.transfer_functions.ensure(
                    names[0],
                    self.get_array_range(
                        representation.node_id,
                        representation.output_port,
                        names[0],
                        association,
                    ),
                )
        value["properties"] = properties
        self._set_representation_state(
            representation_id,
            value,
        )
        self._update_representation(representation_id)
        self._notify_render()

    def set_representation_property(
        self,
        representation_id: str,
        name: str,
        value,
    ) -> None:
        """Set one renderer-agnostic serialized representation property.

        The manager deliberately does not whitelist consumer-specific keys. A
        backend reads the properties it understands and ignores the rest.
        """
        state_value = dict(self.state.representations[representation_id])
        properties = dict(DEFAULT_REPRESENTATION_PROPERTIES)
        properties.update(state_value.get("properties", {}))
        if name == "color_by" and value is not None:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("color_by must be null or [array_name, association]")
            array_name, association = str(value[0]), str(value[1])
            if association not in {"point", "cell"}:
                raise ValueError(f"Unknown array association: {association}")
            representation = self.get_representation(representation_id)
            self.transfer_functions.ensure(
                array_name,
                self.get_array_range(
                    representation.node_id,
                    representation.output_port,
                    array_name,
                    association,
                ),
            )
            value = [array_name, association]

        properties[str(name)] = value
        state_value["properties"] = properties
        self._set_representation_state(representation_id, state_value)
        self._update_representation(representation_id)
        self._notify_render()

    def refresh_representation(self, representation_id: str) -> None:
        self._update_representation(representation_id)
        self._notify_render()

    def discover_transfer_functions(self, node_id: str) -> None:
        self.transfer_functions.discover_node_outputs(node_id)

    # -------------------------------------------------------------------------
    # Output data
    # -------------------------------------------------------------------------

    def get_arrays(
        self,
        node_id: str,
        output_port: int,
    ) -> dict[str, list[str]]:
        processor = self.pipeline.processor(node_id)
        data = processor.GetOutputDataObject(output_port)
        result = {
            "point": [],
            "cell": [],
        }

        if data is None:
            return result

        point_data = data.GetPointData()
        for i in range(point_data.GetNumberOfArrays()):
            name = point_data.GetArrayName(i)
            if name:
                result["point"].append(name)

        cell_data = data.GetCellData()
        for i in range(cell_data.GetNumberOfArrays()):
            name = cell_data.GetArrayName(i)
            if name:
                result["cell"].append(name)

        return result

    def get_array_range(
        self,
        node_id: str,
        output_port: int,
        array_name: str,
        association: str = "point",
    ) -> tuple[float, float] | None:
        processor = self.pipeline.processor(node_id)
        data = processor.GetOutputDataObject(output_port)
        if data is None:
            return None

        attributes = (
            data.GetPointData() if association == "point" else data.GetCellData()
        )
        array = attributes.GetArray(array_name)
        if array is None:
            return None

        component = -1 if array.GetNumberOfComponents() > 1 else 0
        minimum, maximum = array.GetRange(component)
        return (float(minimum), float(maximum))

    def get_global_array_range(self, array_name: str) -> tuple[float, float] | None:
        minimum = None
        maximum = None
        for node_id, node in self.pipeline.nodes.items():
            for output_port in range(node.processor.GetNumberOfOutputPorts()):
                arrays = self.get_arrays(node_id, output_port)
                for association in ("point", "cell"):
                    if array_name not in arrays[association]:
                        continue
                    data_range = self.get_array_range(
                        node_id, output_port, array_name, association
                    )
                    if data_range is None:
                        continue
                    minimum = (
                        data_range[0]
                        if minimum is None
                        else min(minimum, data_range[0])
                    )
                    maximum = (
                        data_range[1]
                        if maximum is None
                        else max(maximum, data_range[1])
                    )
        if minimum is None or maximum is None:
            return None
        return (float(minimum), float(maximum))

    # -------------------------------------------------------------------------
    # View properties
    # -------------------------------------------------------------------------

    def get_view_property(self, view_id: str, name: str):
        self.get_view(view_id)
        property_state = self.state.views[view_id].get("properties", {}).get(name)
        if property_state is None:
            raise ValueError(f"Unknown view property: {name}")
        return deepcopy(property_state.get("value"))

    def set_view_property(
        self,
        view_id: str,
        name: str,
        value,
        *,
        notify: bool = True,
    ) -> None:
        self.get_view(view_id)
        properties = self.state.views[view_id].get("properties", {})
        if name not in properties:
            raise ValueError(f"Unknown view property: {name}")

        if name == "camera":
            camera = dict(self.get_view_property(view_id, "camera") or {})
            camera.update(dict(value or {}))
            value = _normalize_camera(camera)
        elif name in {"background_color", "world_ambient_color"} and not isinstance(
            value, str
        ):
            value = _rgb_to_hex(tuple(map(float, value)))
        elif name == "world_ambient_intensity":
            value = max(0.0, float(value))
        elif name == "fps_limit":
            value = max(1, int(round(float(value))))
        elif name in {"debug", "distributed"}:
            value = bool(value)

        value = deepcopy(value)

        # Update authoritative logical state first. This lets a worker materialize
        # the view immediately when Distributed Rendering transitions to true.
        view = deepcopy(self.state.views[view_id])
        properties = dict(view.get("properties", {}))
        property_state = dict(properties[name])
        property_state["value"] = value
        properties[name] = property_state
        view["properties"] = properties
        self.state.views = {**self.state.views, view_id: view}

        if name == "distributed":
            if self._should_materialize_view(view_id):
                if not self._is_view_materialized(view_id):
                    self._materialize_view(view_id)
            elif self._is_view_materialized(view_id):
                self._dematerialize_view(view_id)

            if self._is_view_materialized(view_id):
                self.frames.set_distributed(view_id, value)
            if notify:
                self._notify_render()
            return

        if name == "debug":
            if self._is_view_materialized(view_id):
                self.frames.set_debug(view_id, value)
        elif name == "fps_limit":
            if self._is_view_materialized(view_id):
                self.frames.set_fps_limit(view_id, float(value))
        elif self._is_view_materialized(view_id):
            backend = self._backend_for_view(view_id)
            backend_id = self.backend_view_id(view_id)
            backend.set_view_property(backend_id, name, value)
            if name == "camera":
                value = _normalize_camera(
                    backend.get_view_property(backend_id, "camera")
                )
                view = deepcopy(self.state.views[view_id])
                properties = dict(view.get("properties", {}))
                property_state = dict(properties[name])
                property_state["value"] = value
                properties[name] = property_state
                view["properties"] = properties
                self.state.views = {**self.state.views, view_id: view}

        if self._is_view_materialized(view_id):
            self.frames.ensure(view_id)
        if notify:
            self._notify_render()

    def reset_camera(
        self,
        view_id: str | None = None,
        *,
        notify: bool = True,
    ) -> None:
        if view_id is None:
            view_id = self.active_view_id
        if view_id is None:
            return
        if not self._is_view_materialized(view_id):
            # Worker-side state-only views receive the resulting camera from rank
            # 0 via the controller's replicated set_view_property call.
            return

        backend = self._backend_for_view(view_id)
        backend_id = self.backend_view_id(view_id)
        backend.reset_camera(backend_id)
        camera = _normalize_camera(backend.get_view_property(backend_id, "camera"))
        view = deepcopy(self.state.views[view_id])
        properties = dict(view.get("properties", {}))
        camera_state = dict(properties["camera"])
        camera_state["value"] = camera
        properties["camera"] = camera_state
        view["properties"] = properties
        self.state.views = {**self.state.views, view_id: view}
        if notify:
            self._notify_render()
        self.frames.ensure(view_id)

    def interact_view_camera(
        self,
        view_id: str,
        mode: str,
        dx: float,
        dy: float,
        viewport_height: float,
    ) -> None:
        """Apply a renderer-agnostic camera gesture to one view."""
        camera = _interacted_camera(
            self.get_view_property(view_id, "camera"),
            mode,
            dx,
            dy,
            viewport_height,
        )
        self.set_view_property(view_id, "camera", camera)

    def set_render_size(self, view_id: str, width: int, height: int) -> None:
        """Apply transient client viewport dimensions without serializing them."""
        width = max(1, int(width))
        height = max(1, int(height))
        new_size = (width, height)
        if self._render_sizes.get(view_id) != new_size:
            self._render_size_revisions[view_id] = (
                self._render_size_revisions.get(view_id, 0) + 1
            )
        self._render_sizes[view_id] = new_size
        revision = self._render_size_revisions.get(view_id, 1)
        if not self._is_view_materialized(view_id):
            return
        self.frames.set_render_size(view_id, width, height, revision=revision)
        backend = self._frame_backend_for_view(view_id)
        if backend.set_render_size(self.backend_view_id(view_id), width, height):
            self.frames.ensure(view_id)

    def prune_render_sizes(self) -> None:
        """Drop transient sizes for logical views that no longer exist."""
        active_ids = set(self.state.views)
        self._render_sizes = {
            view_id: size
            for view_id, size in self._render_sizes.items()
            if view_id in active_ids
        }
        self._render_size_revisions = {
            view_id: revision
            for view_id, revision in self._render_size_revisions.items()
            if view_id in active_ids
        }

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _is_view_materialized(self, view_id: str) -> bool:
        return view_id in self._view_backend_ids

    def _should_materialize_view(self, view_id: str) -> bool:
        if distributed.is_root or not distributed.enabled:
            return True
        return bool(self.get_view_property(view_id, "distributed"))

    def _materialize_view(self, view_id: str) -> None:
        if self._is_view_materialized(view_id):
            return

        value = self.state.views[view_id]
        backend = self.backend_for_type(value["type"])
        backend_id = view_id
        backend.add_view(RenderView(id=backend_id, name=value["name"]))
        self._view_backend_ids[view_id] = backend_id

        # Apply cheap view properties first. Camera is applied after scene
        # representations so clipping/bounds-dependent backend state is valid.
        for property_name in VIEW_PROPERTY_NAMES:
            if property_name in {"camera", "debug", "fps_limit", "distributed"}:
                continue
            backend.set_view_property(
                backend_id,
                property_name,
                deepcopy(value["properties"][property_name]["value"]),
            )

        for representation in self.representations:
            if view_id not in representation.view_ids:
                continue
            if not self.pipeline.has_valid_output(representation.node_id):
                continue
            node = self.pipeline.nodes[representation.node_id]
            backend.add_representation(
                representation,
                RenderView(id=backend_id, name=value["name"]),
                node.processor,
            )

        camera = deepcopy(value["properties"]["camera"]["value"])
        if camera is not None:
            backend.set_view_property(backend_id, "camera", camera)
        camera = _normalize_camera(backend.get_view_property(backend_id, "camera"))
        view = deepcopy(self.state.views[view_id])
        view["properties"]["camera"]["value"] = camera
        self.state.views = {**self.state.views, view_id: view}

        render_size = self._render_sizes.get(view_id)
        if render_size is not None:
            self._frame_backend_for_view(view_id).set_render_size(
                backend_id, *render_size
            )

        self.frames.set_debug(view_id, bool(self.get_view_property(view_id, "debug")))
        self.frames.set_fps_limit(
            view_id, float(self.get_view_property(view_id, "fps_limit"))
        )
        self.frames.set_distributed(
            view_id, bool(self.get_view_property(view_id, "distributed"))
        )
        if render_size is not None:
            self.frames.set_render_size(
                view_id,
                *render_size,
                revision=self._render_size_revisions.get(view_id, 1),
            )
        self.frames.register_view(
            view_id, self._frame_backend_for_view(view_id), backend_id
        )

    def _dematerialize_view(self, view_id: str) -> None:
        if not self._is_view_materialized(view_id):
            return
        backend_id = self._view_backend_ids[view_id]
        backend = self._backend_for_view(view_id)
        self.frames.unregister_view(view_id)
        backend.remove_view(backend_id)
        self._view_backend_ids.pop(view_id, None)

    def _backend_view(self, view_id: str) -> RenderView:
        view = self.get_view(view_id)
        return RenderView(id=self.backend_view_id(view_id), name=view.name)

    def _set_representation_state(
        self,
        representation_id: str,
        value: dict,
    ) -> None:
        representations = dict(self.state.representations)
        representations[representation_id] = value
        self.state.representations = representations

    def _update_representation(self, representation_id: str) -> None:
        representation = self.get_representation(representation_id)
        if not self.pipeline.has_valid_output(representation.node_id):
            return
        node = self.pipeline.nodes[representation.node_id]
        for view_id in tuple(representation.view_ids):
            if not self._is_view_materialized(view_id):
                continue
            self._backend_for_view(view_id).update_representation(
                representation, self._backend_view(view_id), node.processor
            )

    def _notify_render(self) -> None:
        self.frames.ensure_all()

    def backend_for_type(self, view_type: str) -> RenderingBackend:
        backend = self._backends.get(view_type)
        if backend is not None:
            return backend

        if view_type == "mitsuba":
            from vtkweb.rendering.mitsuba_backend import MitsubaRenderingBackend

            backend = MitsubaRenderingBackend(self.transfer_functions.get)
            self._backends[view_type] = backend
            return backend

        raise ValueError(f"Unknown rendering backend: {view_type}")

    def _backend_for_view(self, view_id: str) -> RenderingBackend:
        return self.backend_for_type(self.state.views[view_id]["type"])

    def _frame_backend_for_view(self, view_id: str) -> FrameRenderingBackend:
        backend = self._backend_for_view(view_id)
        if not isinstance(backend, FrameRenderingBackend):
            raise TypeError(f"View is not backed by a frame renderer: {view_id}")
        return backend


def _normalize_camera(value) -> dict:
    value = dict(value or {})
    result = {}
    for name in ("position", "target", "up", "center_of_rotation"):
        if value.get(name) is not None:
            result[name] = [float(component) for component in value[name]]
    if value.get("fov") is not None:
        result["fov"] = float(value["fov"])
    if value.get("parallel_projection") is not None:
        result["parallel_projection"] = bool(value["parallel_projection"])
    if value.get("parallel_scale") is not None:
        result["parallel_scale"] = float(value["parallel_scale"])
    return result


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return vector
    return vector / norm


def _rotate_vector(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalized(axis)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * np.dot(axis, vector) * (1.0 - cosine)
    )


def _interacted_camera(
    camera: dict,
    mode: str,
    dx: float,
    dy: float,
    viewport_height: float,
) -> dict:
    """Return a new renderer-independent camera after one client gesture."""
    result = _normalize_camera(camera)
    position = np.asarray(result.get("position", [0.0, 0.0, 1.0]), dtype=np.float64)
    target = np.asarray(result.get("target", [0.0, 0.0, 0.0]), dtype=np.float64)
    up = _normalized(np.asarray(result.get("up", [0.0, 1.0, 0.0]), dtype=np.float64))
    center = np.asarray(result.get("center_of_rotation", target), dtype=np.float64)
    dx = float(dx)
    dy = float(dy)

    if mode == "orbit":
        offset = position - center
        radians_per_pixel = math.radians(0.35)
        offset = _rotate_vector(offset, up, -dx * radians_per_pixel)
        forward = _normalized(-offset)
        right = np.cross(forward, up)
        if np.linalg.norm(right) > 1.0e-12:
            right = _normalized(right)
            pitch = -dy * radians_per_pixel
            offset = _rotate_vector(offset, right, pitch)
            up = _normalized(_rotate_vector(up, right, pitch))
        position = center + offset
        target = center.copy()

    elif mode == "pan":
        forward_vector = target - position
        distance = max(float(np.linalg.norm(forward_vector)), 1.0e-9)
        forward = _normalized(forward_vector)
        right = np.cross(forward, up)
        if np.linalg.norm(right) > 1.0e-12:
            right = _normalized(right)
            screen_up = _normalized(np.cross(right, forward))
            height = max(float(viewport_height), 1.0)
            if result.get("parallel_projection"):
                world_per_pixel = (
                    2.0 * float(result.get("parallel_scale", 1.0)) / height
                )
            else:
                fov = float(result.get("fov", 30.0))
                world_per_pixel = (
                    2.0 * distance * math.tan(0.5 * math.radians(fov)) / height
                )
            shift = (-dx * right + dy * screen_up) * world_per_pixel
            position += shift
            target += shift
            center += shift
            up = screen_up

    elif mode == "zoom":
        factor = math.exp(max(-4.0, min(4.0, dy * 0.01)))
        if result.get("parallel_projection"):
            result["parallel_scale"] = max(
                1.0e-12,
                min(1.0e12, float(result.get("parallel_scale", 1.0)) * factor),
            )
        else:
            offset = position - target
            distance = float(np.linalg.norm(offset))
            if distance > 1.0e-12:
                next_distance = max(1.0e-9, min(1.0e12, distance * factor))
                position = target + offset * (next_distance / distance)
    else:
        raise ValueError(f"Unknown camera interaction mode: {mode}")

    result["position"] = [float(v) for v in position]
    result["target"] = [float(v) for v in target]
    result["up"] = [float(v) for v in up]
    result["center_of_rotation"] = [float(v) for v in center]
    return result


def _rgb_to_hex(
    color: tuple[float, float, float],
) -> str:
    values = [round(max(0.0, min(1.0, component)) * 255) for component in color]
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"
