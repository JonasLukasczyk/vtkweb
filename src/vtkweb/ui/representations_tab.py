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

    @state.change(
        "active_node_id",
        "active_representation_output_port",
        "pipeline",
        "transfer_functions",
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


def build_representations_tab(
    ctrl,
) -> None:
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
                    classes="vtkweb-compact-select",
                    model_value=("representation.kind",),
                    items=(
                        [
                            {"title": "Surface", "value": "surface"},
                            {"title": "Wireframe", "value": "wireframe"},
                            {"title": "Outline", "value": "outline"},
                            {"title": "Volume", "value": "volume"},
                        ],
                    ),
                    item_title="title",
                    item_value="value",
                    density="compact",
                    variant="plain",
                    hide_details=True,
                    update_modelValue=(
                        ctrl.set_representation_kind,
                        "[representation.id,$event]",
                    ),
                )

            with html.Div(
                v_if=("representation.kind !== 'outline'"),
                classes="mt-1",
            ):
                with html.Div(classes="vtkweb-select-box"):
                    html.Span("Color by", classes="vtkweb-control-label")
                    v3.VSelect(
                        classes="vtkweb-compact-select",
                        model_value=(
                            "representation.properties.color_by === null "
                            "? 'fixed' "
                            ": representation.properties.color_by[1] + ':' + "
                            "representation.properties.color_by[0]",
                        ),
                        items=(
                            "[{ title: 'Fixed', value: 'fixed' }, "
                            "...color_array_items]",
                        ),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        variant="plain",
                        hide_details=True,
                        update_modelValue=(
                            ctrl.set_representation_array,
                            (
                                "[representation.id,"
                                "$event === 'fixed' ? null : $event.split(':').slice(1).join(':'),"
                                "$event === 'fixed' ? 'point' : $event.split(':')[0]]"
                            ),
                        ),
                    )

                with html.Label(
                    v_if=("representation.properties.color_by === null"),
                    classes="vtkweb-color-box mt-1",
                ):
                    html.Span("Color", classes="vtkweb-control-label")
                    html.Input(
                        type="color",
                        value=("representation.properties.color || '#ffffff'",),
                        input=(
                            ctrl.set_representation_property,
                            "[representation.id,'color',$event.target.value]",
                        ),
                    )

            # -------------------------------------------------------------
            # Volume controls
            # -------------------------------------------------------------

            with html.Div(
                v_if=("representation.kind === 'volume'"),
                classes="mt-1",
            ):
                with html.Div(classes="vtkweb-select-box mt-1"):
                    html.Span("Interpolation", classes="vtkweb-control-label")
                    v3.VSelect(
                        classes="vtkweb-compact-select",
                        model_value=("representation.properties.interpolation",),
                        items=(
                            [
                                {"title": "Linear", "value": "linear"},
                                {"title": "Nearest", "value": "nearest"},
                            ],
                        ),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        variant="plain",
                        hide_details=True,
                        update_modelValue=(
                            ctrl.set_representation_property,
                            "[representation.id,'interpolation',$event]",
                        ),
                    )

                with html.Div(classes="vtkweb-select-box mt-1"):
                    html.Span("Blend", classes="vtkweb-control-label")
                    v3.VSelect(
                        classes="vtkweb-compact-select",
                        model_value=("representation.properties.blend_mode",),
                        items=(
                            [
                                {"title": "Composite", "value": "composite"},
                                {"title": "Maximum intensity", "value": "maximum"},
                                {"title": "Minimum intensity", "value": "minimum"},
                            ],
                        ),
                        item_title="title",
                        item_value="value",
                        density="compact",
                        variant="plain",
                        hide_details=True,
                        update_modelValue=(
                            ctrl.set_representation_property,
                            "[representation.id,'blend_mode',$event]",
                        ),
                    )

                with html.Label(classes="vtkweb-bool-row mt-1"):
                    html.Span("Shading", classes="vtkweb-control-label")
                    html.Input(
                        type="checkbox",
                        checked=("Boolean(representation.properties.shade)",),
                        change=(
                            ctrl.set_representation_property,
                            "[representation.id,'shade',$event.target.checked]",
                        ),
                    )

                for label, key, step, minimum, maximum in (
                    ("Ambient", "ambient", "0.05", "0", "1"),
                    ("Diffuse", "diffuse", "0.05", "0", "1"),
                    ("Specular", "specular", "0.05", "0", "1"),
                    ("Specular power", "specular_power", "1", "1", "100"),
                    (
                        "Global illumination",
                        "global_illumination_reach",
                        "0.05",
                        "0",
                        "1",
                    ),
                    ("Scattering", "volumetric_scattering_blending", "0.05", "0", "1"),
                ):
                    with html.Label(classes="vtkweb-input-box mt-1"):
                        html.Span(label, classes="vtkweb-control-label")
                        html.Input(
                            type="number",
                            step=step,
                            min=minimum,
                            max=maximum,
                            value=(f"representation.properties.{key}",),
                            classes="vtkweb-range-input",
                            change=(
                                ctrl.set_representation_property,
                                f"[representation.id,'{key}',Number($event.target.value)]",
                            ),
                        )

                with html.Label(classes="vtkweb-bool-row mt-1"):
                    html.Span("Auto sample distance", classes="vtkweb-control-label")
                    html.Input(
                        type="checkbox",
                        checked=(
                            "Boolean(representation.properties.auto_adjust_sample_distances)",
                        ),
                        change=(
                            ctrl.set_representation_property,
                            "[representation.id,'auto_adjust_sample_distances',$event.target.checked]",
                        ),
                    )

                with html.Div(
                    v_if=("!representation.properties.auto_adjust_sample_distances"),
                    classes="vtkweb-input-box mt-1",
                ):
                    html.Span("Sample distance", classes="vtkweb-control-label")
                    html.Input(
                        type="number",
                        min="0.000001",
                        step="any",
                        value=("representation.properties.sample_distance",),
                        classes="vtkweb-range-input",
                        change=(
                            ctrl.set_representation_property,
                            "[representation.id,'sample_distance',Number($event.target.value)]",
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
            "volume",
        ):
            with v3.VCol(cols=3):
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
