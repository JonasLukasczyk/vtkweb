from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering import RenderManager


def initialize_representations_tab(
    state,
    ctrl,
    pipeline: PipelineGraph,
    rendering: RenderManager,
) -> None:
    state.representations = []

    state.representation_output_ports = []
    state.active_representation_output_port = 0

    state.representation_kind_items = [
        {
            "title": "Surface",
            "value": "surface",
        },
        {
            "title": "Wireframe",
            "value": "wireframe",
        },
        {
            "title": "Outline",
            "value": "outline",
        },
    ]

    state.color_array_items = []

    # -------------------------------------------------------------------------
    # State synchronization
    # -------------------------------------------------------------------------

    def update_representation_state(
        node_id: str | None,
    ) -> None:
        if node_id is None:
            with state:
                state.representations = []
                state.representation_output_ports = []
                state.color_array_items = []

            return

        algorithm = (
            pipeline.nodes[
                node_id
            ].algorithm
        )

        output_count = (
            algorithm.GetNumberOfOutputPorts()
        )

        output_ports = [
            {
                "title": f"Output {port}",
                "value": port,
            }
            for port in range(
                output_count
            )
        ]

        active_port = int(
            state.active_representation_output_port
        )

        if (
            active_port < 0
            or active_port >= output_count
        ):
            active_port = 0

        arrays = (
            rendering.get_arrays(
                node_id,
                active_port,
            )
            if output_count
            else {
                "point": [],
                "cell": [],
            }
        )

        color_items = [
            {
                "title": f"{name} (Point)",
                "value": f"point:{name}",
            }
            for name in arrays["point"]
        ]

        color_items += [
            {
                "title": f"{name} (Cell)",
                "value": f"cell:{name}",
            }
            for name in arrays["cell"]
        ]

        representations = (
            rendering.get_representations(
                node_id,
                active_port,
            )
            if output_count
            else ()
        )

        counts: dict[
            str,
            int,
        ] = {}

        items = []

        active_view_id = (
            rendering.active_view_id
        )

        for representation in representations:
            counts[representation.kind] = (
                counts.get(
                    representation.kind,
                    0,
                )
                + 1
            )

            if representation.array_name is None:
                color_array = None
            else:
                color_array = (
                    f"{representation.association}:"
                    f"{representation.array_name}"
                )

            scalar_range = (
                representation.scalar_range
                or (0.0, 1.0)
            )

            items.append(
                {
                    "id": representation.id,
                    "label": (
                        f"{representation.kind.title()} "
                        f"{counts[representation.kind]}"
                    ),
                    "kind": representation.kind,
                    "color_array": color_array,
                    "range_min": scalar_range[0],
                    "range_max": scalar_range[1],
                    "in_active_view": (
                        active_view_id
                        in representation.view_ids
                    ),
                }
            )

        with state:
            state.representation_output_ports = (
                output_ports
            )

            state.active_representation_output_port = (
                active_port
            )

            state.representations = (
                items
            )

            state.color_array_items = (
                color_items
            )

    # -------------------------------------------------------------------------
    # Output port
    # -------------------------------------------------------------------------

    def set_representation_output_port(
        output_port: int,
    ) -> None:
        state.active_representation_output_port = (
            int(output_port)
        )

        if pipeline.active_node_id is not None:
            update_representation_state(
                pipeline.active_node_id
            )

    # -------------------------------------------------------------------------
    # Representation creation / removal
    # -------------------------------------------------------------------------

    def add_representation(
        kind: str,
    ) -> None:
        node_id = (
            pipeline.active_node_id
        )

        if node_id is None:
            return

        output_port = int(
            state.active_representation_output_port
        )

        rendering.add_representation(
            node_id,
            output_port=output_port,
            kind=kind,
            view_ids={
                rendering.active_view_id
            },
        )

        update_representation_state(
            node_id
        )

        ctrl.update_node_visibility_state()
        ctrl.view_update()

    def remove_representation(
        representation_id: str,
    ) -> None:
        representation = (
            rendering.get_representation(
                representation_id
            )
        )

        node_id = (
            representation.node_id
        )

        rendering.remove_representation(
            representation_id
        )

        update_representation_state(
            node_id
        )

        ctrl.update_node_visibility_state()
        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Active-view membership
    # -------------------------------------------------------------------------

    def toggle_representation_in_active_view(
        representation_id: str,
    ) -> None:
        rendering.toggle_representation_in_view(
            representation_id,
            rendering.active_view_id,
        )

        representation = (
            rendering.get_representation(
                representation_id
            )
        )

        update_representation_state(
            representation.node_id
        )

        ctrl.update_node_visibility_state()
        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Kind
    # -------------------------------------------------------------------------

    def set_representation_kind(
        representation_id: str,
        kind: str,
    ) -> None:
        representation = (
            rendering.get_representation(
                representation_id
            )
        )

        node_id = (
            representation.node_id
        )

        rendering.set_representation_kind(
            representation_id,
            kind,
        )

        update_representation_state(
            node_id
        )

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Coloring
    # -------------------------------------------------------------------------

    def set_color_array(
        representation_id: str,
        value,
    ) -> None:
        representation = (
            rendering.get_representation(
                representation_id
            )
        )

        node_id = (
            representation.node_id
        )

        if not value:
            rendering.set_array(
                representation_id,
                None,
            )

        else:
            association, array_name = (
                value.split(
                    ":",
                    1,
                )
            )

            rendering.set_array(
                representation_id,
                array_name,
                association,
            )

        update_representation_state(
            node_id
        )

        ctrl.view_update()

    def set_color_range_min(
        representation_id: str,
        value,
    ) -> None:
        if value in ("", None):
            return

        representation = (
            rendering.get_representation(
                representation_id
            )
        )

        if representation.scalar_range is None:
            return

        _, maximum = (
            representation.scalar_range
        )

        rendering.set_scalar_range(
            representation_id,
            float(value),
            maximum,
        )

        update_representation_state(
            representation.node_id
        )

        ctrl.view_update()

    def set_color_range_max(
        representation_id: str,
        value,
    ) -> None:
        if value in ("", None):
            return

        representation = (
            rendering.get_representation(
                representation_id
            )
        )

        if representation.scalar_range is None:
            return

        minimum, _ = (
            representation.scalar_range
        )

        rendering.set_scalar_range(
            representation_id,
            minimum,
            float(value),
        )

        update_representation_state(
            representation.node_id
        )

        ctrl.view_update()

    def fit_color_range(
        representation_id: str,
    ) -> None:
        representation = (
            rendering.get_representation(
                representation_id
            )
        )

        if representation.array_name is None:
            return

        scalar_range = (
            rendering.get_array_range(
                representation.node_id,
                representation.output_port,
                representation.array_name,
                representation.association,
            )
        )

        if scalar_range is None:
            return

        rendering.set_scalar_range(
            representation_id,
            *scalar_range,
        )

        update_representation_state(
            representation.node_id
        )

        ctrl.view_update()

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    ctrl.update_representation_state = (
        update_representation_state
    )

    ctrl.set_representation_output_port = (
        set_representation_output_port
    )

    ctrl.add_representation = (
        add_representation
    )

    ctrl.remove_representation = (
        remove_representation
    )

    ctrl.toggle_representation_in_active_view = (
        toggle_representation_in_active_view
    )

    ctrl.set_representation_kind = (
        set_representation_kind
    )

    ctrl.set_color_array = (
        set_color_array
    )

    ctrl.set_color_range_min = (
        set_color_range_min
    )

    ctrl.set_color_range_max = (
        set_color_range_max
    )

    ctrl.fit_color_range = (
        fit_color_range
    )


def build_representations_tab(
    ctrl,
) -> None:
    html.Div(
        "Representations",
        classes="vtkweb-section-title",
    )

    # -------------------------------------------------------------------------
    # Output ports
    # -------------------------------------------------------------------------

    with v3.VTabs(
        model_value=(
            "active_representation_output_port",
        ),
        density="compact",
        grow=True,
        classes="mb-3",
        update_modelValue=(
            ctrl.set_representation_output_port,
            "[$event]",
        ),
    ):
        v3.VTab(
            "{{ output.title }}",
            v_for=(
                "output in "
                "representation_output_ports"
            ),
            key="output.value",
            value=("output.value",),
        )

    # -------------------------------------------------------------------------
    # Existing representations
    # -------------------------------------------------------------------------

    with html.Div(
        classes="vtkweb-representation-cards",
    ):
        with html.Div(
            v_for=(
                "representation "
                "in representations"
            ),
            key="representation.id",
            classes="vtkweb-representation-card",
        ):
            # Header
            with html.Div(
                classes="vtkweb-representation-header",
            ):
                html.Span(
                    "{{ representation.label }}",
                    classes="vtkweb-representation-title",
                )

                with v3.VBtn(
                    icon=True,
                    size="x-small",
                    variant="text",
                    title="Toggle representation in active render view",
                    click=(
                        ctrl.toggle_representation_in_active_view,
                        "[representation.id]",
                    ),
                ):
                    v3.VIcon(
                        icon=(
                            "representation.in_active_view "
                            "? 'mdi-eye' "
                            ": 'mdi-eye-off'"
                        ),
                    )

                html.Button(
                    "×",
                    type="button",
                    title="Remove representation",
                    classes="vtkweb-representation-remove",
                    click=(
                        ctrl.remove_representation,
                        "[representation.id]",
                    ),
                )

            # Type
            with html.Div(
                classes="vtkweb-select-box",
            ):
                html.Span(
                    "Type",
                    classes="vtkweb-control-label",
                )

                v3.VSelect(
                    items=(
                        "representation_kind_items",
                    ),
                    model_value=(
                        "representation.kind",
                    ),
                    density="compact",
                    hide_details=True,
                    variant="plain",
                    classes="vtkweb-select-control",
                    update_modelValue=(
                        ctrl.set_representation_kind,
                        (
                            "[representation.id,"
                            "$event]"
                        ),
                    ),
                )

            # Coloring
            with html.Div(
                v_if=(
                    "representation.kind "
                    "!== 'outline'"
                ),
                classes="mt-1",
            ):
                with html.Div(
                    classes="vtkweb-select-box",
                ):
                    html.Span(
                        "Color by",
                        classes="vtkweb-control-label",
                    )

                    v3.VSelect(
                        items=(
                            "color_array_items",
                        ),
                        model_value=(
                            "representation.color_array",
                        ),
                        clearable=True,
                        density="compact",
                        hide_details=True,
                        variant="plain",
                        classes="vtkweb-select-control",
                        update_modelValue=(
                            ctrl.set_color_array,
                            (
                                "[representation.id,"
                                "$event]"
                            ),
                        ),
                    )

                with html.Div(
                    v_if=(
                        "representation.color_array "
                        "!== null"
                    ),
                    classes="vtkweb-range-row",
                ):
                    html.Input(
                        type="number",
                        step="any",
                        value=(
                            "representation.range_min",
                        ),
                        classes="vtkweb-range-input",
                        change=(
                            ctrl.set_color_range_min,
                            (
                                "[representation.id,"
                                "$event.target.value]"
                            ),
                        ),
                    )

                    html.Input(
                        type="number",
                        step="any",
                        value=(
                            "representation.range_max",
                        ),
                        classes="vtkweb-range-input",
                        change=(
                            ctrl.set_color_range_max,
                            (
                                "[representation.id,"
                                "$event.target.value]"
                            ),
                        ),
                    )

                    v3.VBtn(
                        "Fit",
                        size="small",
                        click=(
                            ctrl.fit_color_range,
                            "[representation.id]",
                        ),
                    )

    # -------------------------------------------------------------------------
    # Add representation
    # -------------------------------------------------------------------------

    with v3.VRow(
        dense=True,
        classes="mt-3",
    ):
        for kind in (
            "surface",
            "wireframe",
            "outline",
        ):
            with v3.VCol(
                cols=4
            ):
                v3.VBtn(
                    kind.title(),
                    block=True,
                    size="small",
                    click=(
                        ctrl.add_representation,
                        f"['{kind}']",
                    ),
                )
