from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from vtkweb.pipeline import PipelineGraph
from vtkweb.rendering import RenderManager
from vtkweb.rendering.base import (
    REPRESENTATION_PROPERTY_GROUPS,
    REPRESENTATION_PROPERTY_SPECS,
)

def _kind_condition(spec) -> str:
    kinds = spec.get("kinds")
    if not kinds:
        return "true"
    return " || ".join(
        f"representation.kind === '{kind}'"
        for kind in sorted(kinds)
    )


def _group_condition(group_name: str) -> str:
    kinds = set()
    unrestricted = False
    for spec in REPRESENTATION_PROPERTY_SPECS.values():
        if spec.get("group") != group_name:
            continue
        if "kinds" not in spec:
            unrestricted = True
            break
        kinds.update(spec["kinds"])
    if unrestricted:
        return "true"
    return " || ".join(
        f"representation.kind === '{kind}'"
        for kind in sorted(kinds)
    ) or "false"


def _render_representation_property(ctrl, key: str) -> None:
    spec = REPRESENTATION_PROPERTY_SPECS[key]
    condition = _kind_condition(spec)
    kind = spec["kind"]

    if kind == "array":
        with html.Div(v_if=(condition,), classes="vtkweb-select-box"):
            html.Span(spec["label"], classes="vtkweb-control-label")
            v3.VSelect(
                classes="vtkweb-compact-select",
                model_value=(
                    "representation.properties.color_by === null "
                    "? 'fixed' "
                    ": representation.properties.color_by[1] + ':' + "
                    "representation.properties.color_by[0]",
                ),
                items=("[{ title: 'Fixed', value: 'fixed' }, ...color_array_items]",),
                item_title="title",
                item_value="value",
                density="compact",
                variant="plain",
                hide_details=True,
                update_modelValue=(
                    ctrl.set_representation_property,
                    (
                        "[representation.id,'color_by',"
                        "$event === 'fixed' ? null : "
                        "[$event.split(':').slice(1).join(':'), $event.split(':')[0]]]"
                    ),
                ),
            )
        return

    if kind == "color":
        with html.Label(
            v_if=(f"({condition}) && representation.properties.color_by === null",),
            classes="vtkweb-color-box",
        ):
            html.Span(spec["label"], classes="vtkweb-control-label")
            html.Input(
                type="color",
                value=(f"representation.properties.{key} || '#ffffff'",),
                input=(
                    ctrl.set_representation_property,
                    f"[representation.id,'{key}',$event.target.value]",
                ),
            )
        return

    if kind == "choice":
        items = [
            {"title": title, "value": value}
            for title, value in spec.get("options", ())
        ]
        with html.Div(v_if=(condition,), classes="vtkweb-select-box"):
            html.Span(spec["label"], classes="vtkweb-control-label")
            v3.VSelect(
                classes="vtkweb-compact-select",
                model_value=(f"representation.properties.{key}",),
                items=(items,),
                item_title="title",
                item_value="value",
                density="compact",
                variant="plain",
                hide_details=True,
                update_modelValue=(
                    ctrl.set_representation_property,
                    f"[representation.id,'{key}',$event]",
                ),
            )
        return

    if kind == "bool":
        with html.Label(v_if=(condition,), classes="vtkweb-bool-row"):
            html.Span(spec["label"], classes="vtkweb-control-label")
            html.Input(
                type="checkbox",
                checked=(f"Boolean(representation.properties.{key})",),
                change=(
                    ctrl.set_representation_property,
                    f"[representation.id,'{key}',$event.target.checked]",
                ),
            )
        return

    if kind in {"int", "float"}:
        visible = condition
        with html.Label(v_if=(visible,), classes="vtkweb-input-box"):
            html.Span(spec["label"], classes="vtkweb-control-label")
            html.Input(
                type="number",
                min=str(spec.get("min", "")),
                max=str(spec.get("max", "")),
                step=str(spec.get("step", 1 if kind == "int" else "any")),
                value=(f"representation.properties.{key}",),
                change=(
                    ctrl.set_representation_property,
                    f"[representation.id,'{key}',Number($event.target.value)]",
                ),
            )


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

            # -------------------------------------------------------------
            # Grouped representation properties. Group membership, labels,
            # icons, ordering, and initial expansion all come from the schema.
            # -------------------------------------------------------------

            for group_name, group in REPRESENTATION_PROPERTY_GROUPS.items():
                with html.Details(
                    v_if=(_group_condition(group_name),),
                    classes="vtkweb-property-group",
                    raw_attrs=["open"] if group.get("expanded") else [],
                ):
                    with html.Summary(classes="vtkweb-property-group-header"):
                        v3.VIcon(
                            group.get("icon", "mdi-tune"),
                            size="small",
                            classes="vtkweb-property-group-icon",
                        )
                        html.Span(group["label"])

                    with html.Div(classes="vtkweb-property-group-body"):
                        for key, spec in REPRESENTATION_PROPERTY_SPECS.items():
                            if spec.get("group") == group_name:
                                _render_representation_property(ctrl, key)

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
                            f"'{kind}', [active_view_id], 1]"
                        ),
                    ),
                )
