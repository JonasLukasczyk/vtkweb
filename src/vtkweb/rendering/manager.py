from __future__ import annotations

from collections.abc import Iterable

from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering.base import (
    REPRESENTATION_KINDS,
    RenderView,
    RenderingBackend,
    Representation,
)
from vtkweb.rendering.vtk_backend import (
    VTKRenderingBackend,
)


class RenderManager:
    def __init__(
        self,
        pipeline: PipelineGraph,
        backend: RenderingBackend | None = None,
    ) -> None:
        self.pipeline = pipeline

        self.backend = (
            backend
            or VTKRenderingBackend()
        )

        self._views: dict[
            str,
            RenderView,
        ] = {}

        self._representations: dict[
            str,
            Representation,
        ] = {}

        # ---------------------------------------------------------------------
        # Initial view
        # ---------------------------------------------------------------------

        view = self.add_view(
            "View 1"
        )

        self._active_view_id = (
            view.id
        )

        # ---------------------------------------------------------------------
        # Initial representations
        # ---------------------------------------------------------------------

        for node in pipeline.nodes.values():
            view_ids = (
                {view.id}
                if node.visible
                else set()
            )

            self.add_representation(
                node.id,
                output_port=0,
                kind="surface",
                view_ids=view_ids,
            )

        self.reset_camera(
            view.id
        )

    # -------------------------------------------------------------------------
    # Views
    # -------------------------------------------------------------------------

    @property
    def views(
        self,
    ) -> tuple[
        RenderView,
        ...,
    ]:
        return tuple(
            self._views.values()
        )

    @property
    def active_view_id(
        self,
    ) -> str:
        return self._active_view_id

    @property
    def active_view(
        self,
    ) -> RenderView:
        return self.get_view(
            self._active_view_id
        )

    def get_view(
        self,
        view_id: str,
    ) -> RenderView:
        return self._views[
            view_id
        ]

    def add_view(
        self,
        name: str | None = None,
    ) -> RenderView:
        if name is None:
            name = (
                f"View {len(self._views) + 1}"
            )

        view = RenderView(
            name=name
        )

        self._views[
            view.id
        ] = view

        self.backend.add_view(
            view
        )

        return view

    def remove_view(
        self,
        view_id: str,
    ) -> None:
        self.get_view(
            view_id
        )

        # First remove the concrete backend objects.
        for representation in tuple(
            self._representations.values()
        ):
            if view_id in representation.view_ids:
                self.unassign_representation(
                    representation.id,
                    view_id,
                )

        self.backend.remove_view(
            view_id
        )

        del self._views[
            view_id
        ]

        if (
            self._active_view_id
            == view_id
        ):
            self._active_view_id = next(
                iter(self._views),
                "",
            )

    def set_active_view(
        self,
        view_id: str,
    ) -> None:
        self.get_view(
            view_id
        )

        self._active_view_id = (
            view_id
        )

    # -------------------------------------------------------------------------
    # Representations
    # -------------------------------------------------------------------------

    @property
    def representations(
        self,
    ) -> tuple[
        Representation,
        ...,
    ]:
        return tuple(
            self._representations.values()
        )

    def get_representation(
        self,
        representation_id: str,
    ) -> Representation:
        return self._representations[
            representation_id
        ]

    def get_representations(
        self,
        node_id: str,
        output_port: int | None = None,
    ) -> tuple[
        Representation,
        ...,
    ]:
        return tuple(
            representation
            for representation
            in self._representations.values()
            if (
                representation.node_id
                == node_id
                and (
                    output_port is None
                    or representation.output_port
                    == output_port
                )
            )
        )

    def add_representation(
        self,
        node_id: str,
        *,
        output_port: int = 0,
        kind: str = "surface",
        view_ids: Iterable[str] = (),
    ) -> Representation:
        if kind not in REPRESENTATION_KINDS:
            raise ValueError(
                f"Unknown representation kind: {kind}"
            )

        node = self.pipeline.nodes[
            node_id
        ]

        output_count = (
            node.algorithm
            .GetNumberOfOutputPorts()
        )

        if (
            output_port < 0
            or output_port >= output_count
        ):
            raise ValueError(
                f"{node.name} has "
                f"{output_count} output port(s); "
                f"port {output_port} is invalid"
            )

        representation = Representation(
            node_id=node_id,
            output_port=output_port,
            kind=kind,
        )

        self._representations[
            representation.id
        ] = representation

        for view_id in view_ids:
            self.assign_representation(
                representation.id,
                view_id,
            )

        return representation

    def remove_representation(
        self,
        representation_id: str,
    ) -> None:
        representation = (
            self.get_representation(
                representation_id
            )
        )

        # Important:
        # remove all concrete backend instances first.
        #
        # Using unassign_representation() here means there
        # is exactly one code path for removing a
        # representation from a view.
        for view_id in tuple(
            representation.view_ids
        ):
            self.unassign_representation(
                representation_id,
                view_id,
            )

        # Only destroy the abstract representation after
        # every view assignment is gone.
        del self._representations[
            representation_id
        ]

    def remove_node(
        self,
        node_id: str,
    ) -> None:
        for representation in list(
            self.get_representations(
                node_id
            )
        ):
            self.remove_representation(
                representation.id
            )

    # -------------------------------------------------------------------------
    # View assignment
    # -------------------------------------------------------------------------

    def representation_in_view(
        self,
        representation_id: str,
        view_id: str,
    ) -> bool:
        representation = (
            self.get_representation(
                representation_id
            )
        )

        return (
            view_id
            in representation.view_ids
        )

    def assign_representation(
        self,
        representation_id: str,
        view_id: str,
    ) -> None:
        representation = (
            self.get_representation(
                representation_id
            )
        )

        if (
            view_id
            in representation.view_ids
        ):
            return

        view = self.get_view(
            view_id
        )

        node = self.pipeline.nodes[
            representation.node_id
        ]

        # Create the concrete backend representation
        # before recording the assignment.
        self.backend.add_representation(
            representation,
            view,
            node.algorithm,
        )

        representation.view_ids.add(
            view_id
        )

    def unassign_representation(
        self,
        representation_id: str,
        view_id: str,
    ) -> None:
        representation = (
            self.get_representation(
                representation_id
            )
        )

        if (
            view_id
            not in representation.view_ids
        ):
            return

        # Remove the backend object first.
        self.backend.remove_representation(
            representation.id,
            view_id,
        )

        # Then update our abstract model.
        representation.view_ids.remove(
            view_id
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

    # -------------------------------------------------------------------------
    # Node visibility relative to a view
    # -------------------------------------------------------------------------

    def node_visible_in_view(
        self,
        node_id: str,
        view_id: str,
    ) -> bool:
        return any(
            view_id
            in representation.view_ids
            for representation
            in self.get_representations(
                node_id
            )
        )

    def toggle_node_in_view(
        self,
        node_id: str,
        view_id: str,
    ) -> None:
        representations = list(
            self.get_representations(
                node_id
            )
        )

        if not representations:
            representation = (
                self.add_representation(
                    node_id,
                    output_port=0,
                )
            )

            self.assign_representation(
                representation.id,
                view_id,
            )

            return

        if self.node_visible_in_view(
            node_id,
            view_id,
        ):
            for representation in representations:
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
            raise ValueError(
                f"Unknown representation kind: {kind}"
            )

        representation = (
            self.get_representation(
                representation_id
            )
        )

        representation.kind = (
            kind
        )

        self._update_representation(
            representation
        )

    # -------------------------------------------------------------------------
    # Coloring
    # -------------------------------------------------------------------------

    def set_array(
        self,
        representation_id: str,
        array_name: str | None,
        association: str = "point",
    ) -> None:
        representation = (
            self.get_representation(
                representation_id
            )
        )

        representation.array_name = (
            array_name
        )

        representation.association = (
            association
        )

        if array_name is None:
            representation.scalar_range = (
                None
            )
        else:
            representation.scalar_range = (
                self.get_array_range(
                    representation.node_id,
                    representation.output_port,
                    array_name,
                    association,
                )
            )

        self._update_representation(
            representation
        )

    def set_scalar_range(
        self,
        representation_id: str,
        minimum: float,
        maximum: float,
    ) -> None:
        representation = (
            self.get_representation(
                representation_id
            )
        )

        representation.scalar_range = (
            float(minimum),
            float(maximum),
        )

        self._update_representation(
            representation
        )

    # -------------------------------------------------------------------------
    # Output data
    # -------------------------------------------------------------------------

    def get_arrays(
        self,
        node_id: str,
        output_port: int,
    ) -> dict[
        str,
        list[str],
    ]:
        algorithm = (
            self.pipeline.nodes[
                node_id
            ].algorithm
        )

        algorithm.Update()

        data = (
            algorithm.GetOutputDataObject(
                output_port
            )
        )

        result = {
            "point": [],
            "cell": [],
        }

        if data is None:
            return result

        point_data = (
            data.GetPointData()
        )

        for i in range(
            point_data.GetNumberOfArrays()
        ):
            name = (
                point_data.GetArrayName(i)
            )

            if name:
                result[
                    "point"
                ].append(name)

        cell_data = (
            data.GetCellData()
        )

        for i in range(
            cell_data.GetNumberOfArrays()
        ):
            name = (
                cell_data.GetArrayName(i)
            )

            if name:
                result[
                    "cell"
                ].append(name)

        return result

    def get_array_range(
        self,
        node_id: str,
        output_port: int,
        array_name: str,
        association: str = "point",
    ) -> tuple[
        float,
        float,
    ] | None:
        algorithm = (
            self.pipeline.nodes[
                node_id
            ].algorithm
        )

        algorithm.Update()

        data = (
            algorithm.GetOutputDataObject(
                output_port
            )
        )

        if data is None:
            return None

        attributes = (
            data.GetPointData()
            if association == "point"
            else data.GetCellData()
        )

        array = attributes.GetArray(
            array_name
        )

        if array is None:
            return None

        minimum, maximum = (
            array.GetRange()
        )

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
        color: tuple[
            float,
            float,
            float,
        ],
    ) -> None:
        view = self.get_view(
            view_id
        )

        view.settings.background_color = (
            color
        )

        self.backend.set_view_settings(
            view
        )

    def reset_camera(
        self,
        view_id: str | None = None,
    ) -> None:
        if view_id is None:
            view_id = (
                self.active_view_id
            )

        self.backend.reset_camera(
            view_id
        )

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _update_representation(
        self,
        representation: Representation,
    ) -> None:
        node = self.pipeline.nodes[
            representation.node_id
        ]

        for view_id in tuple(
            representation.view_ids
        ):
            self.backend.update_representation(
                representation,
                self.get_view(
                    view_id
                ),
                node.algorithm,
            )

    def output_visible_in_view(
        self,
        node_id: str,
        output_port: int,
        view_id: str,
    ) -> bool:
        return any(
            view_id in representation.view_ids
            for representation
            in self.get_representations(
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

        # If the output has no representation yet,
        # create the default surface representation.
        if not representations:
            representation = (
                self.add_representation(
                    node_id,
                    output_port=output_port,
                    kind="surface",
                )
            )

            self.assign_representation(
                representation.id,
                view_id,
            )

            return

        # Treat the output as visible if at least one of
        # its representations belongs to this view.
        visible = any(
            view_id in representation.view_ids
            for representation in representations
        )

        if visible:
            for representation in representations:
                if (
                    view_id
                    in representation.view_ids
                ):
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
