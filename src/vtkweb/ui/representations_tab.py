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
    # ``state.representations`` is the global authoritative representation
    # model owned by RenderManager. The only local state here is inspector
    # selection plus derived color-array choices for the selected output.
    state.active_representation_output_port = 0
    state.color_array_items = []

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

    def update_representation_state(
        node_id: str | None,
    ) -> None:
        if node_id is None:
            state.color_array_items = []
            return

        node = pipeline.nodes[node_id]
        output_count = node.algorithm.GetNumberOfOutputPorts()

        active_port = int(state.active_representation_output_port)

        if output_count == 0 or active_port < 0 or active_port >= output_count:
            active_port = 0
            state.active_representation_output_port = 0

        if output_count == 0:
            state.color_array_items = []
            return

        arrays = rendering.get_arrays(
            node_id,
            active_port,
        )

        state.color_array_items = [
            {
                "title": f"{name} (Point)",
                "value": f"point:{name}",
            }
            for name in arrays["point"]
        ] + [
            {
                "title": f"{name} (Cell)",
                "value": f"cell:{name}",
            }
            for name in arrays["cell"]
        ]

    def set_representation_output_port(
        output_port: int,
    ) -> None:
        state.active_representation_output_port = int(output_port)

        if pipeline.active_node_id is not None:
            update_representation_state(pipeline.active_node_id)

    def add_representation(
        kind: str,
    ) -> None:
        node_id = pipeline.active_node_id
        if node_id is None:
            return

        rendering.add_representation(
            node_id,
            output_port=int(state.active_representation_output_port),
            kind=kind,
            view_ids={rendering.active_view_id},
        )
        ctrl.view_update()

    def remove_representation(
        representation_id: str,
    ) -> None:
        rendering.remove_representation(representation_id)
        ctrl.view_update()

    def toggle_representation_in_active_view(
        representation_id: str,
    ) -> None:
        rendering.toggle_representation_in_view(
            representation_id,
            rendering.active_view_id,
        )
        ctrl.view_update()

    def set_representation_kind(
        representation_id: str,
        kind: str,
    ) -> None:
        rendering.set_representation_kind(
            representation_id,
            kind,
        )
        ctrl.view_update()

    def set_color_array(
        representation_id: str,
        value,
    ) -> None:
        if not value:
            rendering.set_array(
                representation_id,
                None,
            )
        else:
            association, array_name = value.split(
                ":",
                1,
            )
            rendering.set_array(
                representation_id,
                array_name,
                association,
            )

        ctrl.view_update()

    def set_color_range_min(
        representation_id: str,
        value,
    ) -> None:
        if value in ("", None):
            return

        representation = rendering.get_representation(representation_id)
        if representation.scalar_range is None:
            return

        _, maximum = representation.scalar_range
        rendering.set_scalar_range(
            representation_id,
            float(value),
            maximum,
        )
        ctrl.view_update()

    def set_color_range_max(
        representation_id: str,
        value,
    ) -> None:
        if value in ("", None):
            return

        representation = rendering.get_representation(representation_id)
        if representation.scalar_range is None:
            return

        minimum, _ = representation.scalar_range
        rendering.set_scalar_range(
            representation_id,
            minimum,
            float(value),
        )
        ctrl.view_update()

    def fit_color_range(
        representation_id: str,
    ) -> None:
        representation = rendering.get_representation(representation_id)

        if representation.array_name is None:
            return

        scalar_range = rendering.get_array_range(
            representation.node_id,
            representation.output_port,
            representation.array_name,
            representation.association,
        )
        if scalar_range is None:
            return

        rendering.set_scalar_range(
            representation_id,
            *scalar_range,
        )
        ctrl.view_update()

    ctrl.update_representation_state = update_representation_state
    ctrl.set_representation_output_port = set_representation_output_port
    ctrl.add_representation = add_representation
    ctrl.remove_representation = remove_representation
    ctrl.toggle_representation_in_active_view = toggle_representation_in_active_view
    ctrl.set_representation_kind = set_representation_kind
    ctrl.set_color_array = set_color_array
    ctrl.set_color_range_min = set_color_range_min
    ctrl.set_color_range_max = set_color_range_max
    ctrl.fit_color_range = fit_color_range


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
        model_value=("active_representation_output_port",),
        density="compact",
        grow=True,
        classes="mb-3",
        update_modelValue=(
            ctrl.set_representation_output_port,
            "[$event]",
        ),
    ):
        v3.VTab(
            "Output {{ port - 1 }}",
            v_for=("port in (pipeline.nodes[active_node_id]?.output_port_count || 0)"),
            key=("port - 1",),
            value=("port - 1",),
        )

    # -------------------------------------------------------------------------
    # Representations for active node/output
    # -------------------------------------------------------------------------

    with html.Div(
        classes="vtkweb-representation-cards",
    ):
        with html.Div(
            v_for=(
                "representation in Object.values(representations).filter("
                "rep => rep.node_id === active_node_id && "
                "rep.output_port === active_representation_output_port)"
            ),
            key=("representation.id",),
            classes="vtkweb-representation-card",
        ):
            with html.Div(
                classes="vtkweb-representation-header",
            ):
                html.Span(
                    (
                        "{{ representation.kind.charAt(0).toUpperCase() + "
                        "representation.kind.slice(1) }}"
                    ),
                    classes="vtkweb-representation-title",
                )

                with html.Button(
                    type="button",
                    title=("Toggle representation in active render view"),
                    click=(
                        ctrl.toggle_representation_in_active_view,
                        "[representation.id]",
                    ),
                    style=(
                        "position:relative;"
                        "display:flex;"
                        "align-items:center;"
                        "justify-content:center;"
                        "width:28px;"
                        "height:28px;"
                        "padding:0;"
                        "border:0;"
                        "background:transparent;"
                        "color:inherit;"
                        "cursor:pointer;"
                    ),
                ):
                    html.Span(
                        "👁",
                        style=("font-size:15px;line-height:1;user-select:none;"),
                    )
                    html.Span(
                        "",
                        v_if=("!representation.view_ids.includes(active_view_id)"),
                        style=(
                            "position:absolute;"
                            "left:5px;"
                            "top:13px;"
                            "width:18px;"
                            "height:2px;"
                            "background:currentColor;"
                            "transform:rotate(-45deg);"
                            "transform-origin:center;"
                            "pointer-events:none;"
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

            with html.Div(
                classes="vtkweb-select-box",
            ):
                html.Span(
                    "Type",
                    classes="vtkweb-control-label",
                )
                v3.VSelect(
                    items=("representation_kind_items",),
                    model_value=("representation.kind",),
                    density="compact",
                    hide_details=True,
                    variant="plain",
                    classes="vtkweb-select-control",
                    update_modelValue=(
                        ctrl.set_representation_kind,
                        ("[representation.id,$event]"),
                    ),
                )

            with html.Div(
                v_if=("representation.kind !== 'outline'"),
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
                        items=("color_array_items",),
                        model_value=(
                            "representation.array_name "
                            "? representation.association + ':' + "
                            "representation.array_name : null",
                        ),
                        clearable=True,
                        density="compact",
                        hide_details=True,
                        variant="plain",
                        classes="vtkweb-select-control",
                        update_modelValue=(
                            ctrl.set_color_array,
                            ("[representation.id,$event]"),
                        ),
                    )

                with html.Div(
                    v_if=("representation.array_name !== null"),
                    classes="vtkweb-range-row",
                ):
                    html.Input(
                        type="number",
                        step="any",
                        value=("representation.scalar_range?.[0] ?? 0",),
                        classes="vtkweb-range-input",
                        change=(
                            ctrl.set_color_range_min,
                            ("[representation.id,$event.target.value]"),
                        ),
                    )
                    html.Input(
                        type="number",
                        step="any",
                        value=("representation.scalar_range?.[1] ?? 1",),
                        classes="vtkweb-range-input",
                        change=(
                            ctrl.set_color_range_max,
                            ("[representation.id,$event.target.value]"),
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
            with v3.VCol(cols=4):
                v3.VBtn(
                    kind.title(),
                    block=True,
                    size="small",
                    click=(
                        ctrl.add_representation,
                        f"['{kind}']",
                    ),
                )
