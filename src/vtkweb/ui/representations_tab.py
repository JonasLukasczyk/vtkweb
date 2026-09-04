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
    state.active_representation_output_port = 0
    state.color_array_items = []
    state.representation_kind_items = [
        {"title": "Surface", "value": "surface"},
        {"title": "Wireframe", "value": "wireframe"},
        {"title": "Outline", "value": "outline"},
    ]

    @state.change(
        "active_node_id",
        "active_representation_output_port",
        "pipeline",
    )
    def update_color_array_items(**_):
        node_id = pipeline.active_node_id
        if node_id is None or node_id not in pipeline.nodes:
            state.color_array_items = []
            return

        node = pipeline.nodes[node_id]
        output_count = node.processor.GetNumberOfOutputPorts()
        output_port = int(state.active_representation_output_port)

        if output_count == 0:
            state.active_representation_output_port = 0
            state.color_array_items = []
            return

        if output_port < 0 or output_port >= output_count:
            state.active_representation_output_port = 0
            output_port = 0

        arrays = rendering.get_arrays(node_id, output_port)
        state.color_array_items = [
            {"title": f"{name} (Point)", "value": f"point:{name}"}
            for name in arrays["point"]
        ] + [
            {"title": f"{name} (Cell)", "value": f"cell:{name}"}
            for name in arrays["cell"]
        ]

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
        if scalar_range is not None:
            ctrl.set_representation_scalar_range(
                representation_id,
                *scalar_range,
            )

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
        v_model=("active_representation_output_port", 0),
        density="compact",
        grow=True,
        classes="mb-3",
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
                        ctrl.toggle_representation_in_view,
                        "[representation.id,active_view_id]",
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
                            ctrl.set_representation_array,
                            (
                                "[representation.id,"
                                "$event ? $event.split(':')[1] : null,"
                                "$event ? $event.split(':')[0] : 'point']"
                            ),
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
                            ctrl.set_representation_scalar_range,
                            (
                                "[representation.id,$event.target.value,"
                                "representation.scalar_range[1]]"
                            ),
                        ),
                    )
                    html.Input(
                        type="number",
                        step="any",
                        value=("representation.scalar_range?.[1] ?? 1",),
                        classes="vtkweb-range-input",
                        change=(
                            ctrl.set_representation_scalar_range,
                            (
                                "[representation.id,representation.scalar_range[0],"
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
            with v3.VCol(cols=4):
                v3.VBtn(
                    kind.title(),
                    block=True,
                    size="small",
                    click=(
                        ctrl.add_representation,
                        (
                            "[active_node_id, "
                            "active_representation_output_port, "
                            f"'{kind}', [active_view_id]]"
                        ),
                    ),
                )
