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

        view = self.add_view("View 1")
        self.set_active_view(view.id)

        self.reset_camera(view.id)

    # -------------------------------------------------------------------------
    # Views
    # -------------------------------------------------------------------------

    @property
    def views(
        self,
    ) -> tuple[RenderView, ...]:
        return tuple(self.get_view(view_id) for view_id in self.state.views)

    @property
    def active_view_id(
        self,
    ) -> str:
        return self.state.active_view_id

    @property
    def active_view(
        self,
    ) -> RenderView:
        return self.get_view(self.active_view_id)

    def get_view(
        self,
        view_id: str,
    ) -> RenderView:
        value = self.state.views[view_id]
        return RenderView(
            id=value["id"],
            name=value["name"],
            settings=ViewSettings(
                background_color=_hex_to_rgb(value["background_color"])
            ),
        )

    def add_view(
        self,
        name: str | None = None,
        *,
        view_id: str | None = None,
    ) -> RenderView:
        if name is None:
            name = f"View {len(self.state.views) + 1}"

        view_id = view_id or uuid4().hex
        if view_id in self.state.views:
            raise ValueError(f"View ID already exists: {view_id}")

        value = {
            "id": view_id,
            "name": name,
            "background_color": "#1a1a1a",
        }

        views = dict(self.state.views)
        views[view_id] = value
        self.state.views = views

        view = self.get_view(view_id)
        self.backend.add_view(view)
        return view

    def remove_view(
        self,
        view_id: str,
    ) -> None:
        self.get_view(view_id)

        for representation in tuple(self.representations):
            if view_id in representation.view_ids:
                self.unassign_representation(
                    representation.id,
                    view_id,
                )

        self.backend.remove_view(view_id)

        views = dict(self.state.views)
        del views[view_id]
        self.state.views = views

        if self.active_view_id == view_id:
            self.state.active_view_id = next(
                iter(views),
                None,
            )

    def rename_view(
        self,
        view_id: str,
        new_view_id: str,
        *,
        name: str | None = None,
    ) -> RenderView:
        self.get_view(view_id)

        if new_view_id != view_id and new_view_id in self.state.views:
            raise ValueError(f"View ID already exists: {new_view_id}")

        value = dict(self.state.views[view_id])
        value["id"] = new_view_id
        if name is not None:
            value["name"] = name

        views = dict(self.state.views)
        del views[view_id]
        views[new_view_id] = value

        self.backend.rename_view(
            view_id,
            new_view_id,
        )

        representations = dict(self.state.representations)
        for representation_id, representation in representations.items():
            if view_id not in representation.get("view_ids", []):
                continue

            updated = dict(representation)
            updated["view_ids"] = [
                new_view_id if item == view_id else item
                for item in representation.get("view_ids", [])
            ]
            representations[representation_id] = updated

        self.state.views = views
        self.state.representations = representations

        if self.active_view_id == view_id:
            self.state.active_view_id = new_view_id

        return self.get_view(new_view_id)

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
            view,
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
            view_id,
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

    def toggle_representation_in_view(
        self,
        representation_id: str,
        view_id: str,
    ) -> None:
        if self.representation_in_view(
            representation_id,
            view_id,
        ):
            self.unassign_representation(
                representation_id,
                view_id,
            )
        else:
            self.assign_representation(
                representation_id,
                view_id,
            )

    def output_visible_in_view(
        self,
        node_id: str,
        output_port: int,
        view_id: str,
    ) -> bool:
        return any(
            view_id in representation.view_ids
            for representation in self.get_representations(
                node_id,
                output_port,
            )
        )

    def toggle_output_in_view(
        self,
        node_id: str,
        output_port: int,
        view_id: str,
    ) -> None:
        representations = list(
            self.get_representations(
                node_id,
                output_port,
            )
        )

        if not representations:
            representation = self.add_representation(
                node_id,
                output_port=output_port,
                kind="surface",
            )
            self.assign_representation(
                representation.id,
                view_id,
            )
            return

        visible = any(
            view_id in representation.view_ids for representation in representations
        )

        if visible:
            for representation in representations:
                if view_id in representation.view_ids:
                    self.unassign_representation(
                        representation.id,
                        view_id,
                    )
        else:
            for representation in representations:
                self.assign_representation(
                    representation.id,
                    view_id,
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

        self.backend.set_view_settings(self.get_view(view_id))

    def reset_camera(
        self,
        view_id: str | None = None,
    ) -> None:
        if view_id is None:
            view_id = self.active_view_id

        self.backend.reset_camera(view_id)

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

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
                self.get_view(view_id),
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
