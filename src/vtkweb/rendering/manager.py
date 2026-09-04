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
        self.backend = backend or VTKRenderingBackend()

        self.state.views = {}
        self.state.representations = {}
        self.state.active_view_id = None

        # VtkLocalView components are created once when the Trame UI is built.
        # Keep a small pool of backend render windows alive and map logical
        # vtk views onto those slots. Logical view IDs remain fully dynamic and
        # serializable while the client-side VTK components stay stable.
        self._slot_ids = tuple(f"vtk_slot_{index}" for index in range(8))
        self._slot_owners: dict[str, str | None] = {slot: None for slot in self._slot_ids}
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
        return self.get_view(view_id)

    def remove_view(
        self,
        view_id: str,
    ) -> None:
        self.get_view(view_id)

        for representation in tuple(self.representations):
            if view_id in representation.view_ids:
                self.unassign_representation(representation.id, view_id)

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

        scalar_range = value.get("scalar_range")

        return Representation(
            id=value["id"],
            node_id=value["node_id"],
            output_port=int(value["output_port"]),
            kind=value["kind"],
            array_name=value.get("array_name"),
            association=value.get(
                "association",
                "point",
            ),
            scalar_range=(tuple(scalar_range) if scalar_range is not None else None),
            color=value.get("color", "#ffffff"),
            view_ids=set(
                value.get(
                    "view_ids",
                    [],
                )
            ),
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
        if representation_id in self.state.representations:
            raise ValueError(f"Representation ID already exists: {representation_id}")

        value = {
            "id": representation_id,
            "node_id": node_id,
            "output_port": int(output_port),
            "kind": kind,
            "array_name": None,
            "association": "point",
            "scalar_range": None,
            "color": "#ffffff",
            "view_ids": [],
        }

        representations = dict(self.state.representations)
        representations[representation_id] = value
        self.state.representations = representations

        for view_id in view_ids:
            self.assign_representation(
                representation_id,
                view_id,
            )

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
            )

        representations = dict(self.state.representations)
        representations.pop(
            representation_id,
            None,
        )
        self.state.representations = representations

    def remove_node(
        self,
        node_id: str,
    ) -> None:
        for representation in list(self.get_representations(node_id)):
            self.remove_representation(representation.id)

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

    def unassign_representation(
        self,
        representation_id: str,
        view_id: str,
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
        self._set_representation_state(
            representation_id,
            value,
        )
        self._update_representation(representation_id)

    def set_array(
        self,
        representation_id: str,
        array_name: str | None,
        association: str = "point",
    ) -> None:
        representation = self.get_representation(representation_id)

        scalar_range = None
        if array_name is not None:
            scalar_range = self.get_array_range(
                representation.node_id,
                representation.output_port,
                array_name,
                association,
            )

        value = dict(self.state.representations[representation_id])
        value.update(
            {
                "array_name": array_name,
                "association": association,
                "scalar_range": (
                    list(scalar_range) if scalar_range is not None else None
                ),
            }
        )
        self._set_representation_state(
            representation_id,
            value,
        )
        self._update_representation(representation_id)

    def set_color(
        self,
        representation_id: str,
        color: str,
    ) -> None:
        value = dict(self.state.representations[representation_id])
        value["color"] = color
        self._set_representation_state(
            representation_id,
            value,
        )
        self._update_representation(representation_id)

    def set_scalar_range(
        self,
        representation_id: str,
        minimum: float,
        maximum: float,
    ) -> None:
        value = dict(self.state.representations[representation_id])
        value["scalar_range"] = [
            float(minimum),
            float(maximum),
        ]
        self._set_representation_state(
            representation_id,
            value,
        )
        self._update_representation(representation_id)

    # -------------------------------------------------------------------------
    # Output data
    # -------------------------------------------------------------------------

    def get_arrays(
        self,
        node_id: str,
        output_port: int,
    ) -> dict[str, list[str]]:
        processor = self.pipeline.processor(node_id)
        processor.Update()

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
        processor.Update()

        data = processor.GetOutputDataObject(output_port)
        if data is None:
            return None

        attributes = (
            data.GetPointData() if association == "point" else data.GetCellData()
        )
        array = attributes.GetArray(array_name)
        if array is None:
            return None

        minimum, maximum = array.GetRange()
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

    def reset_camera(
        self,
        view_id: str | None = None,
    ) -> None:
        if view_id is None:
            view_id = self.active_view_id

        self.backend.reset_camera(self.backend_view_id(view_id))

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
