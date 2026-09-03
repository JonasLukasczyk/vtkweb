from __future__ import annotations

from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering.base import (
    REPRESENTATION_KINDS,
    RenderingBackend,
    Representation,
    ViewSettings,
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

        self.view_settings = ViewSettings()

        self._representations: dict[
            str,
            Representation,
        ] = {}

        self.backend.set_view_settings(
            self.view_settings
        )

        for node in pipeline.nodes.values():
            self.add_representation(
                node.id,
                kind="surface",
                visible=node.visible,
            )

        self.backend.reset_camera()

    # -------------------------------------------------------------------------
    # Representations
    # -------------------------------------------------------------------------

    @property
    def representations(
        self,
    ) -> tuple[Representation, ...]:
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
    ) -> tuple[Representation, ...]:
        return tuple(
            representation
            for representation
            in self._representations.values()
            if representation.node_id == node_id
        )

    def add_representation(
        self,
        node_id: str,
        *,
        kind: str = "surface",
        visible: bool = True,
    ) -> Representation:
        if kind not in REPRESENTATION_KINDS:
            raise ValueError(
                f"Unknown representation kind: {kind}"
            )

        representation = Representation(
            node_id=node_id,
            kind=kind,
            visible=bool(visible),
        )

        self._representations[
            representation.id
        ] = representation

        node = self.pipeline.nodes[
            node_id
        ]

        self.backend.add_representation(
            representation,
            node.algorithm,
        )

        self._sync_node_visibility(
            node_id
        )

        return representation

    def remove_representation(
        self,
        representation_id: str,
    ) -> None:
        representation = (
            self._representations.pop(
                representation_id
            )
        )

        self.backend.remove_representation(
            representation_id
        )

        self._sync_node_visibility(
            representation.node_id
        )

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

        representation.kind = kind

        self._update_representation(
            representation
        )

    def set_representation_visibility(
        self,
        representation_id: str,
        visible: bool,
    ) -> None:
        representation = (
            self.get_representation(
                representation_id
            )
        )

        representation.visible = bool(
            visible
        )

        self._update_representation(
            representation
        )

        self._sync_node_visibility(
            representation.node_id
        )

    # -------------------------------------------------------------------------
    # Node visibility
    # -------------------------------------------------------------------------

    def node_visible(
        self,
        node_id: str,
    ) -> bool:
        return any(
            representation.visible
            for representation
            in self.get_representations(
                node_id
            )
        )

    def toggle_node_visibility(
        self,
        node_id: str,
    ) -> None:
        representations = list(
            self.get_representations(
                node_id
            )
        )

        if not representations:
            self.add_representation(
                node_id,
                visible=True,
            )
            return

        visible = not self.node_visible(
            node_id
        )

        for representation in representations:
            representation.visible = (
                visible
            )

            self._update_representation(
                representation
            )

        self._sync_node_visibility(
            node_id
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
            representation.scalar_range = None

        else:
            representation.scalar_range = (
                self.get_array_range(
                    representation.node_id,
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
    # Arrays
    # -------------------------------------------------------------------------

    def get_arrays(
        self,
        node_id: str,
    ) -> dict[str, list[str]]:
        algorithm = (
            self.pipeline.nodes[
                node_id
            ].algorithm
        )

        algorithm.Update()

        data = algorithm.GetOutputDataObject(
            0
        )

        result = {
            "point": [],
            "cell": [],
        }

        if data is None:
            return result

        point_data = data.GetPointData()

        for i in range(
            point_data.GetNumberOfArrays()
        ):
            name = point_data.GetArrayName(i)

            if name:
                result["point"].append(
                    name
                )

        cell_data = data.GetCellData()

        for i in range(
            cell_data.GetNumberOfArrays()
        ):
            name = cell_data.GetArrayName(i)

            if name:
                result["cell"].append(
                    name
                )

        return result

    def get_array_range(
        self,
        node_id: str,
        array_name: str,
        association: str = "point",
    ) -> tuple[float, float] | None:
        algorithm = (
            self.pipeline.nodes[
                node_id
            ].algorithm
        )

        algorithm.Update()

        data = algorithm.GetOutputDataObject(
            0
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

        minimum, maximum = array.GetRange()

        return (
            float(minimum),
            float(maximum),
        )

    # -------------------------------------------------------------------------
    # View
    # -------------------------------------------------------------------------

    def set_background_color(
        self,
        color: tuple[
            float,
            float,
            float,
        ],
    ) -> None:
        self.view_settings.background_color = (
            color
        )

        self.backend.set_view_settings(
            self.view_settings
        )

    def reset_camera(self) -> None:
        self.backend.reset_camera()

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

        self.backend.update_representation(
            representation,
            node.algorithm,
        )

    def _sync_node_visibility(
        self,
        node_id: str,
    ) -> None:
        if node_id not in self.pipeline.nodes:
            return

        # This is only aggregate state for the
        # pipeline-node eye indicator.
        self.pipeline.nodes[
            node_id
        ].visible = self.node_visible(
            node_id
        )
