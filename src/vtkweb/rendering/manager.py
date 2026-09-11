from __future__ import annotations

from collections.abc import Iterable
import inspect
from uuid import uuid4

from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering.base import (
    REPRESENTATION_KINDS,
    RenderView,
    ProgressiveRenderingBackend,
    RenderingBackend,
    Representation,
)
from vtkweb.rendering.progressive import ProgressiveRenderManager
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
        self.progressive = ProgressiveRenderManager(frame_transport)
        self.transfer_functions = TransferFunctionManager(state, self)
        self.backend = backend or VTKRenderingBackend(self.transfer_functions.get)
        self._backends: dict[str, RenderingBackend] = {"vtk": self.backend}

        self.state.views = {}
        self.state.representations = {}
        self.state.active_view_id = None

        # Monotonic notification used by VtkLocalView adapters. Backend-only
        # representation refreshes do not otherwise mutate Trame state, so the
        # client would have no reason to pull the updated render window.
        self.state.render_revision = 0
        self.state.camera_revision = 0

        # VtkLocalView components are created once when the Trame UI is built.
        # Keep a small pool of backend render windows alive and map logical
        # vtk views onto those slots. Logical view IDs remain fully dynamic and
        # serializable while the client-side VTK components stay stable.
        self._slot_ids = tuple(f"vtk_slot_{index}" for index in range(8))
        self._slot_owners: dict[str, str | None] = {
            slot: None for slot in self._slot_ids
        }
        for slot_id in self._slot_ids:
            self.backend.add_view(RenderView(id=slot_id, name=slot_id))

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
    def backend_slots(self) -> tuple[str, ...]:
        return self._slot_ids

    @property
    def active_view_id(
        self,
    ) -> str | None:
        return self.state.active_view_id

    @property
    def active_view(
        self,
    ) -> RenderView | None:
        if self.active_view_id is None:
            return None
        return self.get_view(self.active_view_id)

    def get_view(self, view_id: str) -> RenderView:
        value = self.state.views[view_id]
        if value.get("type") not in {"vtk", "mitsuba"}:
            raise ValueError(f"View is not a render view: {view_id}")
        return RenderView(id=value["id"], name=value["name"])

    def backend_view_id(self, view_id: str) -> str:
        value = self.state.views[view_id]
        if value.get("type") not in {"vtk", "mitsuba"}:
            raise ValueError(f"View is not a render view: {view_id}")
        return value["backend_id"]

    def get_render_window(self, view_id: str):
        if self.state.views[view_id].get("type") != "vtk":
            raise ValueError(f"View is not a VTK view: {view_id}")
        return self.backend.get_render_window(self.backend_view_id(view_id))

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

        if view_type == "vtk":
            backend_id = next(
                (slot for slot, owner in self._slot_owners.items() if owner is None),
                None,
            )
            if backend_id is None:
                raise RuntimeError(
                    f"Maximum number of VTK views reached ({len(self._slot_ids)})"
                )
        else:
            backend_id = view_id

        value = {
            "id": view_id,
            "type": view_type,
            "name": name,
            "background_color": "#1a1a1a",
            "world_ambient_color": "#ffffff",
            "world_ambient_intensity": 1.0,
            "backend_id": backend_id,
        }

        self.state.views = {**self.state.views, view_id: value}
        if view_type == "vtk":
            self._slot_owners[backend_id] = view_id
        else:
            self._backend_for_type(view_type).add_view(self._backend_view(view_id))
        backend = self._backend_for_view(view_id)
        for name in ("background_color", "world_ambient_color", "world_ambient_intensity"):
            backend.set_view_property(backend_id, name, self._backend_view_property(name, value[name]))
        if view_type == "mitsuba":
            self.progressive.register_view(
                view_id, self._progressive_backend_for_view(view_id), backend_id
            )
        self._notify_render()
        return self.get_view(view_id)

    def remove_view(
        self,
        view_id: str,
    ) -> None:
        self.get_view(view_id)

        for representation in tuple(self.representations):
            if view_id in representation.view_ids:
                self.unassign_representation(representation.id, view_id, notify=False)

        view_type = self.state.views[view_id]["type"]
        backend_id = self.backend_view_id(view_id)
        if view_type == "vtk":
            self._slot_owners[backend_id] = None
        else:
            self.progressive.unregister_view(view_id)
            self._backend_for_view(view_id).remove_view(backend_id)

        views = dict(self.state.views)
        del views[view_id]
        self.state.views = views


        if self.active_view_id == view_id:
            self.state.active_view_id = next(
                (view.id for view in self.views),
                None,
            )

        self._notify_render()

    def set_active_view(
        self,
        view_id: str,
    ) -> None:
        self.get_view(view_id)
        self.state.active_view_id = view_id

    def switch_view_type(self, view_id: str, view_type: str) -> None:
        """Replace a render backend in place while preserving view state and visibility."""
        if view_type not in {"vtk", "mitsuba"}:
            raise ValueError(f"Unknown render view type: {view_type}")
        value = dict(self.state.views[view_id])
        if value.get("type") == view_type:
            return

        camera = self.get_view_property(view_id, "camera")
        representation_ids = [
            rep.id for rep in self.representations if view_id in rep.view_ids
        ]
        old_type, old_backend_id = value["type"], value["backend_id"]
        old_backend = self._backend_for_type(old_type)
        for representation_id in representation_ids:
            old_backend.remove_representation(representation_id, old_backend_id)

        if old_type == "vtk":
            self._slot_owners[old_backend_id] = None
        else:
            self.progressive.unregister_view(view_id)
            old_backend.remove_view(old_backend_id)

        if view_type == "vtk":
            backend_id = next((slot for slot, owner in self._slot_owners.items() if owner is None), None)
            if backend_id is None:
                raise RuntimeError(f"Maximum number of VTK views reached ({len(self._slot_ids)})")
            self._slot_owners[backend_id] = view_id
        else:
            backend_id = view_id

        value.update(type=view_type, backend_id=backend_id)
        self.state.views = {**self.state.views, view_id: value}
        backend = self._backend_for_type(view_type)
        if view_type == "mitsuba":
            backend.add_view(self._backend_view(view_id))

        for name in ("background_color", "world_ambient_color", "world_ambient_intensity"):
            backend.set_view_property(backend_id, name, self._backend_view_property(name, value[name]))
        backend.set_camera_state(backend_id, camera)

        for representation_id in representation_ids:
            representation = self.get_representation(representation_id)
            if self.pipeline.has_valid_output(representation.node_id):
                backend.add_representation(
                    representation, self._backend_view(view_id),
                    self.pipeline.nodes[representation.node_id].processor,
                )

        if view_type == "mitsuba":
            self.progressive.register_view(
                view_id, self._progressive_backend_for_view(view_id), backend_id
            )
            self.progressive.ensure(view_id)
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
        properties.update(value.get("properties", {}))

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

        self.state.representations = {**self.state.representations, representation_id: value}

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
            representation = self.add_representation(
                node_id, output_port=output_port, kind="outline",
                view_ids=view_ids, notify=False,
            )
            created.append(representation.id)

        return tuple(created)

    def refresh_node(
        self,
        node_id: str,
    ) -> None:
        """Refresh every render representation backed by *node_id*.

        This operation changes backend VTK objects without necessarily changing
        serialized representation state. ``render_revision`` therefore changes
        after the backend is current so VtkLocalView pushes the new scene to the
        browser immediately rather than waiting for a camera interaction.
        """

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
        if self.pipeline.has_valid_output(representation.node_id):
            node = self.pipeline.nodes[representation.node_id]
            self._backend_for_view(view_id).add_representation(
                representation, self._backend_view(view_id), node.processor
            )
            if self._is_mitsuba_view(view_id):
                self.progressive.ensure(view_id)

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
        properties[str(name)] = value
        state_value["properties"] = properties
        self._set_representation_state(representation_id, state_value)
        self._update_representation(representation_id)
        self._notify_render()

    def set_array(
        self,
        representation_id: str,
        array_name: str | None,
        association: str = "point",
    ) -> None:
        """Set renderer-independent ``color_by`` selection for a representation."""
        if association not in {"point", "cell"}:
            raise ValueError(f"Unknown array association: {association}")

        representation = self.get_representation(representation_id)
        color_by = None
        if array_name is not None:
            color_by = [str(array_name), association]
            self.transfer_functions.ensure(
                str(array_name),
                self.get_array_range(
                    representation.node_id,
                    representation.output_port,
                    str(array_name),
                    association,
                ),
            )

        state_value = dict(self.state.representations[representation_id])
        properties = dict(DEFAULT_REPRESENTATION_PROPERTIES)
        properties.update(state_value.get("properties", {}))
        properties["color_by"] = color_by
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

    def get_global_array_range(
        self, array_name: str
    ) -> tuple[float, float] | None:
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
                        data_range[0] if minimum is None else min(minimum, data_range[0])
                    )
                    maximum = (
                        data_range[1] if maximum is None else max(maximum, data_range[1])
                    )
        if minimum is None or maximum is None:
            return None
        return (float(minimum), float(maximum))

    # -------------------------------------------------------------------------
    # View properties
    # -------------------------------------------------------------------------

    def get_view_property(self, view_id: str, name: str):
        self.get_view(view_id)
        if name == "camera":
            backend = self._backend_for_view(view_id)
            return backend.get_camera_state(self.backend_view_id(view_id))
        return self.state.views[view_id].get(name)

    def set_view_property(self, view_id: str, name: str, value) -> None:
        self.get_view(view_id)
        backend = self._backend_for_view(view_id)
        backend_id = self.backend_view_id(view_id)

        if name == "camera":
            backend.set_camera_state(backend_id, dict(value))
            self._notify_camera()
            if self._is_mitsuba_view(view_id):
                self.progressive.ensure(view_id)
            return

        if name in {"background_color", "world_ambient_color"} and not isinstance(value, str):
            value = _rgb_to_hex(tuple(map(float, value)))
        elif name == "world_ambient_intensity":
            value = max(0.0, float(value))
        view = dict(self.state.views[view_id], **{name: value})
        self.state.views = {**self.state.views, view_id: view}
        backend.set_view_property(backend_id, name, self._backend_view_property(name, value))
        if self._is_mitsuba_view(view_id):
            self.progressive.ensure(view_id)
        self._notify_render()

    def reset_camera(self, view_id: str | None = None) -> None:
        if view_id is None:
            view_id = self.active_view_id
        if view_id is None:
            return

        print(
            f"[camera] RenderManager.reset_camera(view_id={view_id}, "
            f"caller={inspect.stack()[1].function})", flush=True
        )
        self._backend_for_view(view_id).reset_camera(self.backend_view_id(view_id))
        self._notify_camera()
        if self._is_mitsuba_view(view_id):
            self.progressive.ensure(view_id)

    def sync_vtk_camera(self, view_id: str, value: dict) -> None:
        """Mirror the client-side VTK camera into the server vtkCamera."""
        if self.state.views.get(view_id, {}).get("type") != "vtk":
            return
        self._backend_for_view(view_id).set_camera_state(
            self.backend_view_id(view_id),
            {
                "position": value.get("position"),
                "target": value.get("target", value.get("focalPoint")),
                "up": value.get("up", value.get("viewUp")),
                "fov": value.get("fov", value.get("viewAngle")),
                "parallel_projection": value.get(
                    "parallel_projection", value.get("parallelProjection")
                ),
                "parallel_scale": value.get(
                    "parallel_scale", value.get("parallelScale")
                ),
            },
        )

    def interact_mitsuba_camera(
        self,
        view_id: str,
        mode: str,
        dx: float,
        dy: float,
        viewport_height: float,
    ) -> None:
        """Apply a transient camera interaction without mutating app state."""
        if not self._is_mitsuba_view(view_id):
            return
        self._progressive_backend_for_view(view_id).interact_camera(
            self.backend_view_id(view_id), mode, dx, dy, viewport_height
        )
        self.progressive.ensure(view_id)

    def set_mitsuba_render_size(self, view_id: str, width: int, height: int) -> None:
        """Apply transient client viewport dimensions without mutating app state."""
        if not self._is_mitsuba_view(view_id):
            return
        backend = self._progressive_backend_for_view(view_id)
        if backend.set_render_size(self.backend_view_id(view_id), width, height):
            self.progressive.ensure(view_id)

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _backend_view(self, view_id: str) -> RenderView:
        view = self.get_view(view_id)
        return RenderView(id=self.backend_view_id(view_id), name=view.name)

    @staticmethod
    def _backend_view_property(name: str, value):
        return _hex_to_rgb(value) if name in {"background_color", "world_ambient_color"} else value

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
            self._backend_for_view(view_id).update_representation(
                representation, self._backend_view(view_id), node.processor
            )

    def _notify_render(self) -> None:
        """Notify VTK clients and ensure progressive views are rendering."""
        self.state.render_revision = int(self.state.render_revision or 0) + 1
        self.progressive.ensure_all()

    def _notify_camera(self) -> None:
        self.state.camera_revision = int(self.state.camera_revision or 0) + 1

    def _backend_for_type(self, view_type: str) -> RenderingBackend:
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
        return self._backend_for_type(self.state.views[view_id]["type"])

    def _progressive_backend_for_view(
        self, view_id: str
    ) -> ProgressiveRenderingBackend:
        backend = self._backend_for_view(view_id)
        if not isinstance(backend, ProgressiveRenderingBackend):
            raise TypeError(f"View is not backed by a progressive renderer: {view_id}")
        return backend

    def _is_mitsuba_view(self, view_id: str) -> bool:
        value = self.state.views.get(view_id)
        return value is not None and value.get("type") == "mitsuba"



def _rgb_to_hex(
    color: tuple[float, float, float],
) -> str:
    values = [round(max(0.0, min(1.0, component)) * 255) for component in color]
    return f"#{values[0]:02x}{values[1]:02x}{values[2]:02x}"


def _hex_to_rgb(
    value: str,
) -> tuple[float, float, float]:
    value = value.lstrip("#")
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
    )
