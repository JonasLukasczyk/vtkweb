from __future__ import annotations

from trame.widgets import client, html
from trame.widgets import vuetify3 as v3
from trame_client.widgets.core import HtmlElement


class _SvgTag(HtmlElement):
    def __init__(self, tag, children=None, **kwargs):
        super().__init__(tag, children, **kwargs)
        self._attr_names += [
            ["classes", "class"],
            ["key", ":key"],
            ["v_for", "v-for"],
            "x",
            "y",
            "x1",
            "y1",
            "x2",
            "y2",
            "width",
            "height",
            "points",
            "cx",
            "cy",
            "r",
        ]


def initialize_transfer_tab(state, ctrl) -> None:
    state.active_transfer_function = None
    state.active_tf_preset = None
    state.transfer_function_items = []

    def update_items(**_):
        names = sorted(state.transfer_functions or {})
        state.transfer_function_items = [
            {"title": name, "value": name} for name in names
        ]
        if not names:
            state.active_transfer_function = None
            state.active_tf_preset = None
        elif state.active_transfer_function not in names:
            state.active_transfer_function = names[0]
            state.active_tf_preset = None

    @state.change("transfer_functions")
    def on_transfer_functions_change(**_):
        update_items()

    def set_active_transfer_function(array_name: str | None) -> None:
        state.active_transfer_function = array_name
        state.active_tf_preset = None

    def apply_active_tf_preset(array_name: str | None, preset_name: str | None) -> None:
        if array_name is None or preset_name is None:
            return
        state.active_tf_preset = preset_name
        ctrl.apply_tf_preset(array_name, preset_name)

    ctrl.set_active_transfer_function = set_active_transfer_function
    ctrl.apply_active_tf_preset = apply_active_tf_preset
    update_items()


def _mapping_range_row(ctrl, mapping_name: str) -> None:
    with html.Div(classes="vtkweb-range-row"):
        html.Input(
            type="number",
            step="any",
            value=(
                f"transfer_functions[active_transfer_function]?.{mapping_name}?.range?.[0] ?? 0",
            ),
            classes="vtkweb-range-input",
            change=(
                ctrl.set_tf_mapping_range,
                f"[active_transfer_function,'{mapping_name}',Number($event.target.value),"
                f"transfer_functions[active_transfer_function].{mapping_name}.range[1]]",
            ),
        )
        html.Input(
            type="number",
            step="any",
            value=(
                f"transfer_functions[active_transfer_function]?.{mapping_name}?.range?.[1] ?? 1",
            ),
            classes="vtkweb-range-input",
            change=(
                ctrl.set_tf_mapping_range,
                f"[active_transfer_function,'{mapping_name}',"
                f"transfer_functions[active_transfer_function].{mapping_name}.range[0],"
                "Number($event.target.value)]",
            ),
        )
        v3.VBtn(
            "Rescale",
            size="small",
            click=(
                ctrl.rescale_tf_mapping,
                f"[active_transfer_function,'{mapping_name}']",
            ),
        )


def build_transfer_tab(ctrl) -> None:
    client.ClientTriggers(
        mounted=r"""
            window.__vtkwebTfEditorCoords = (svg, event) => {
                const rect = svg.getBoundingClientRect();
                const x = Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(rect.width, 1)));
                const y = Math.max(0, Math.min(1, 1 - (event.clientY - rect.top) / Math.max(rect.height, 1)));
                return [x, y];
            };

            window.__vtkwebAddOpacityPoint = (arrayName, event) => {
                if (!arrayName || event.target !== event.currentTarget) return;
                const [x, opacity] = window.__vtkwebTfEditorCoords(event.currentTarget, event);
                trigger('add_tf_opacity_control_point', [arrayName, x, opacity]);
            };

            window.__vtkwebStartOpacityDrag = (arrayName, pointIndex, event) => {
                if (!arrayName || event.button !== 0) return;
                event.preventDefault();
                event.stopPropagation();
                trigger('set_tf_interacting', [true]);
                const svg = event.currentTarget.ownerSVGElement;
                if (!svg) return;

                const move = (moveEvent) => {
                    const [x, opacity] = window.__vtkwebTfEditorCoords(svg, moveEvent);
                    trigger('set_tf_opacity_control_point', [arrayName, pointIndex, x, opacity]);
                };
                const up = () => {
                    window.removeEventListener('pointermove', move);
                    window.removeEventListener('pointerup', up);
                    window.removeEventListener('pointercancel', up);
                    trigger('set_tf_interacting', [false]);
                };

                window.addEventListener('pointermove', move);
                window.addEventListener('pointerup', up);
                window.addEventListener('pointercancel', up);
            };
        """,
        before_unmount=r"""
            delete window.__vtkwebTfEditorCoords;
            delete window.__vtkwebAddOpacityPoint;
            delete window.__vtkwebStartOpacityDrag;
        """,
    )

    html.Div(
        "No transfer functions",
        v_if=("transfer_function_items.length === 0"),
        classes="text-medium-emphasis",
    )

    with html.Div(v_if=("transfer_function_items.length > 0")):
        with html.Div(classes="vtkweb-select-box"):
            html.Span("Transfer", classes="vtkweb-control-label")
            v3.VSelect(
                classes="vtkweb-compact-select",
                model_value=("active_transfer_function",),
                items=("transfer_function_items",),
                item_title="title",
                item_value="value",
                density="compact",
                variant="plain",
                hide_details=True,
                update_modelValue=(ctrl.set_active_transfer_function, "[$event]"),
            )

        html.Div("Color", classes="vtkweb-tf-section-title")
        with html.Div(classes="vtkweb-select-box mt-1"):
            html.Span("Preset", classes="vtkweb-control-label")
            v3.VSelect(
                classes="vtkweb-compact-select",
                model_value=("active_tf_preset",),
                items=("tf_preset_items",),
                item_title="title",
                item_value="value",
                item_props=True,
                density="compact",
                variant="plain",
                hide_details=True,
                placeholder="Matplotlib colormap",
                update_modelValue=(
                    ctrl.apply_active_tf_preset,
                    "[active_transfer_function,$event]",
                ),
            )

        _mapping_range_row(ctrl, "color")

        with html.Table(classes="vtkweb-tf-table"):
            with html.Thead():
                with html.Tr():
                    for label in ("X", "R", "G", "B", ""):
                        html.Th(label)
            with html.Tbody():
                with html.Tr(
                    v_for=(
                        "(point,index) in (transfer_functions[active_transfer_function]?.color?.control_points || [])",
                    ),
                    key=("index",),
                ):
                    for component_index in range(4):
                        with html.Td():
                            html.Input(
                                type="number",
                                min="0",
                                max="1",
                                step="0.01",
                                value=(f"point[{component_index}]",),
                                classes="vtkweb-range-input",
                                change=(
                                    ctrl.set_tf_color_control_point_component,
                                    f"[active_transfer_function,index,{component_index},Number($event.target.value)]",
                                ),
                            )
                    with html.Td():
                        v3.VBtn(
                            "×",
                            size="x-small",
                            disabled=(
                                "transfer_functions[active_transfer_function].color.control_points.length <= 2",
                            ),
                            click=(
                                ctrl.remove_tf_color_control_point,
                                "[active_transfer_function,index]",
                            ),
                        )

        v3.VBtn(
            "Add color point",
            classes="mt-1",
            size="small",
            click=(ctrl.add_tf_color_control_point, "[active_transfer_function]"),
        )

        html.Div("Opacity", classes="vtkweb-tf-section-title")
        _mapping_range_row(ctrl, "opacity")
        html.Div(
            "Click to add a point. Drag points to edit opacity; endpoint x positions stay fixed.",
            classes="vtkweb-tf-help",
        )

        with html.Svg(
            classes="vtkweb-opacity-editor",
            raw_attrs=[
                'viewBox="0 0 300 140"',
                'preserveAspectRatio="none"',
                '@click="window.__vtkwebAddOpacityPoint(active_transfer_function, $event)"',
            ],
        ):
            _SvgTag(
                "rect",
                x="0",
                y="0",
                width="300",
                height="140",
                classes="vtkweb-opacity-bg",
            )
            _SvgTag(
                "line",
                x1="0",
                y1="70",
                x2="300",
                y2="70",
                classes="vtkweb-opacity-grid",
            )
            _SvgTag(
                "line",
                x1="150",
                y1="0",
                x2="150",
                y2="140",
                classes="vtkweb-opacity-grid",
            )
            _SvgTag(
                "polyline",
                points=(
                    "(transfer_functions[active_transfer_function]?.opacity?.control_points || [])"
                    ".map(p => `${p[0] * 300},${(1 - p[1]) * 140}`).join(' ')",
                ),
                classes="vtkweb-opacity-line",
            )
            _SvgTag(
                "circle",
                v_for=(
                    "(point,index) in (transfer_functions[active_transfer_function]?.opacity?.control_points || [])",
                ),
                key=("index",),
                cx=("point[0] * 300",),
                cy=("(1 - point[1]) * 140",),
                r="5",
                classes="vtkweb-opacity-point",
                raw_attrs=[
                    '@pointerdown="window.__vtkwebStartOpacityDrag(active_transfer_function, index, $event)"',
                    "@click.stop",
                    "@dblclick.stop=\"trigger('remove_tf_opacity_control_point', [active_transfer_function,index])\"",
                ],
            )

        html.Div(
            "Double-click an interior opacity point to remove it.",
            classes="vtkweb-tf-help",
        )
