from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering.base import (
    REPRESENTATION_KINDS,
    RenderView,
    RenderingBackend,
    Representation,
    ViewSettings,
)
from vtkweb.rendering.vtk_backend import (
    VTKRenderingBackend,
)


DEFAULT_REPRESENTATION_PROPERTIES = {
    "color_by": None,
    "color": "#ffffff",
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

LEGACY_PROPERTY_NAMES = {
    "color": "color",
    "volume_interpolation": "interpolation",
    "volume_blend_mode": "blend_mode",
    "volume_shade": "shade",
    "volume_ambient": "ambient",
    "volume_diffuse": "diffuse",
    "volume_specular": "specular",
    "volume_specular_power": "specular_power",
    "volume_global_illumination_reach": "global_illumination_reach",
    "volume_scattering_blending": "volumetric_scattering_blending",
    "volume_auto_adjust_sample_distances": "auto_adjust_sample_distances",
    "volume_sample_distance": "sample_distance",
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
    ) -> None:
        self.state = state
        self.pipeline = pipeline
        self.backend = backend or VTKRenderingBackend(state)

        self.state.views = {}
        self.state.representations = {}
        self.state.active_view_id = None
        # Monotonic notification used by VtkLocalView adapters. Backend-only
        # representation refreshes do not otherwise mutate Trame state, so the
        # client would have no reason to pull the updated render window.
        self.state.render_revision = 0

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
            if value.get("type") == "vtk"
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

    def get_view(
        self,
        view_id: str,
    ) -> RenderView:
        value = self.state.views[view_id]
        if value.get("type") != "vtk":
            raise ValueError(f"View is not a VTK view: {view_id}")
        return RenderView(
            id=value["id"],
            name=value["name"],
            settings=ViewSettings(
                background_color=_hex_to_rgb(value["background_color"])
            ),
        )

    def backend_view_id(self, view_id: str) -> str:
        value = self.state.views[view_id]
        if value.get("type") != "vtk":
            raise ValueError(f"View is not a VTK view: {view_id}")
        return value["backend_id"]

    def get_render_window(self, view_id: str):
        return self.backend.get_render_window(self.backend_view_id(view_id))

    def add_view(
        self,
        name: str | None = None,
        *,
        view_id: str | None = None,
    ) -> RenderView:
        if name is None:
            name = f"View {len(self.views) + 1}"

        view_id = view_id or uuid4().hex
        if view_id in self.state.views:
            raise ValueError(f"View ID already exists: {view_id}")

        backend_id = next(
            (slot for slot, owner in self._slot_owners.items() if owner is None),
            None,
        )
        if backend_id is None:
            raise RuntimeError(
                f"Maximum number of VTK views reached ({len(self._slot_ids)})"
            )

        value = {
            "id": view_id,
            "type": "vtk",
            "name": name,
            "background_color": "#1a1a1a",
            "backend_id": backend_id,
        }

        views = dict(self.state.views)
        views[view_id] = value
        self.state.views = views
        self._slot_owners[backend_id] = view_id
        self.backend.set_view_settings(self._backend_view(view_id))
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

        backend_id = self.backend_view_id(view_id)
        self._slot_owners[backend_id] = None

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

        # Backward compatibility for state created before representation
        # properties were collected under the abstract properties mapping.
        for legacy_name, property_name in LEGACY_PROPERTY_NAMES.items():
            if legacy_name in value and property_name not in value.get(
                "properties", {}
            ):
                properties[property_name] = value[legacy_name]

        # Migrate older scalar-selection state into the compound color_by field.
        if properties.get("color_by") is None:
            old = value.get("properties", {})
            array_name = old.get("scalar_array", value.get("array_name"))
            association = old.get(
                "scalar_association", value.get("association", "point")
            )
            if array_name is not None:
                properties["color_by"] = [array_name, association]
        for obsolete in (
            "scalar_array",
            "scalar_association",
            "scalar_range",
            "scalar_component",
        ):
            properties.pop(obsolete, None)

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
        representation_id: str | None = None,
        notify: bool = True,
    ) -> Representation:
        if kind not in REPRESENTATION_KINDS:
            raise ValueError(f"Unknown representation kind: {kind}")

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

        representations = dict(self.state.representations)
        representations[representation_id] = value
        self.state.representations = representations

        for view_id in view_ids:
            self.assign_representation(
                representation_id,
                view_id,
                notify=False,
            )

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
            if view is not None and view.get("type") == "vtk":
                view_ids = (active_view_id,)

        created = []
        for output_port in range(node.processor.GetNumberOfOutputPorts()):
            if self.get_representations(node_id, output_port):
                continue
            representation = self.add_representation(
                node_id,
                output_port=output_port,
                kind="outline",
                view_ids=view_ids,
                notify=False,
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

        view = self.get_view(view_id)
        node = self.pipeline.nodes[representation.node_id]

        self.backend.add_representation(
            representation,
            self._backend_view(view_id),
            node.processor,
        )

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

        self.backend.remove_representation(
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
        """Set compound color selection as [array_name, association]."""
        if association not in {"point", "cell"}:
            raise ValueError("association must be 'point' or 'cell'")
        state_value = dict(self.state.representations[representation_id])
        properties = dict(DEFAULT_REPRESENTATION_PROPERTIES)
        properties.update(state_value.get("properties", {}))
        properties["color_by"] = (
            None if array_name is None else [str(array_name), association]
        )
        state_value["properties"] = properties
        self._set_representation_state(representation_id, state_value)
        self._update_representation(representation_id)
        self._notify_render()

    def refresh_representation(self, representation_id: str) -> None:
        self._update_representation(representation_id)
        self._notify_render()

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
        component: int | None = None,
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

        if component is None:
            # color_by has no component selector. Multi-component arrays are
            # represented by their vector magnitude in every backend.
            range_component = -1 if array.GetNumberOfComponents() > 1 else 0
            minimum, maximum = array.GetRange(range_component)
        else:
            component = max(0, min(int(component), array.GetNumberOfComponents() - 1))
            minimum, maximum = array.GetRange(component)
        return (
            float(minimum),
            float(maximum),
        )

    # -------------------------------------------------------------------------
    # View properties
    # -------------------------------------------------------------------------

    def set_background_color(
        self,
        view_id: str,
        color: tuple[float, float, float],
    ) -> None:
        value = dict(self.state.views[view_id])
        value["background_color"] = _rgb_to_hex(color)

        views = dict(self.state.views)
        views[view_id] = value
        self.state.views = views

        self.backend.set_view_settings(self._backend_view(view_id))
        self._notify_render()

    def reset_camera(
        self,
        view_id: str | None = None,
    ) -> None:
        if view_id is None:
            view_id = self.active_view_id
        if view_id is None:
            return

        self.backend.reset_camera(self.backend_view_id(view_id))
        self._notify_render()

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _backend_view(self, view_id: str) -> RenderView:
        logical = self.get_view(view_id)
        return RenderView(
            id=self.backend_view_id(view_id),
            name=logical.name,
            settings=logical.settings,
        )

    def _set_representation_state(
        self,
        representation_id: str,
        value: dict,
    ) -> None:
        representations = dict(self.state.representations)
        representations[representation_id] = value
        self.state.representations = representations

    def _update_representation(
        self,
        representation_id: str,
    ) -> None:
        representation = self.get_representation(representation_id)
        node = self.pipeline.nodes[representation.node_id]

        for view_id in tuple(representation.view_ids):
            self.backend.update_representation(
                representation,
                self._backend_view(view_id),
                node.processor,
            )

    def _notify_render(self) -> None:
        """Notify client render views after an atomic backend scene change."""
        self.state.render_revision = int(self.state.render_revision or 0) + 1


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
