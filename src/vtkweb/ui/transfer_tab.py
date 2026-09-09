from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3


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


def build_transfer_tab(ctrl) -> None:
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

        with html.Div(classes="vtkweb-select-box mt-2"):
            html.Span("Preset", classes="vtkweb-control-label")
            v3.VSelect(
                classes="vtkweb-compact-select",
                model_value=("active_tf_preset",),
                items=("tf_preset_items",),
                item_title="title",
                item_value="value",
                density="compact",
                variant="plain",
                hide_details=True,
                placeholder="Matplotlib colormap",
                update_modelValue=(
                    ctrl.apply_active_tf_preset,
                    "[active_transfer_function,$event]",
                ),
            )

        with html.Div(classes="vtkweb-range-row mt-2"):
            html.Input(
                type="number",
                step="any",
                value=(
                    "transfer_functions[active_transfer_function]?.range?.[0] ?? 0",
                ),
                classes="vtkweb-range-input",
                change=(
                    ctrl.set_tf_range,
                    "[active_transfer_function,Number($event.target.value),"
                    "transfer_functions[active_transfer_function].range[1]]",
                ),
            )
            html.Input(
                type="number",
                step="any",
                value=(
                    "transfer_functions[active_transfer_function]?.range?.[1] ?? 1",
                ),
                classes="vtkweb-range-input",
                change=(
                    ctrl.set_tf_range,
                    "[active_transfer_function,"
                    "transfer_functions[active_transfer_function].range[0],"
                    "Number($event.target.value)]",
                ),
            )
            v3.VBtn(
                "Rescale",
                size="small",
                click=(ctrl.rescale_tf, "[active_transfer_function]"),
            )

        with html.Table(
            classes="mt-3",
            style="width:100%;border-collapse:collapse;font-size:12px;",
        ):
            with html.Thead():
                with html.Tr():
                    for label in ("t", "R", "G", "B", "O", ""):
                        html.Th(label, style="padding:2px;text-align:left;")
            with html.Tbody():
                with html.Tr(
                    v_for=(
                        "(point,index) in (transfer_functions[active_transfer_function]?.control_points || [])"
                    ),
                    key=("index",),
                ):
                    with html.Td(style="padding:2px;"):
                        html.Input(
                            type="number",
                            min="0",
                            max="1",
                            step="0.01",
                            value=("point[0]",),
                            classes="vtkweb-range-input",
                            change=(
                                ctrl.set_tf_control_point_component,
                                "[active_transfer_function,index,0,Number($event.target.value)]",
                            ),
                        )
                    for component_index in range(1, 5):
                        with html.Td(style="padding:2px;"):
                            html.Input(
                                type="number",
                                min="0",
                                max="1",
                                step="0.01",
                                value=(f"point[{component_index}]",),
                                classes="vtkweb-range-input",
                                change=(
                                    ctrl.set_tf_control_point_component,
                                    f"[active_transfer_function,index,{component_index},Number($event.target.value)]",
                                ),
                            )
                    with html.Td(style="padding:2px;"):
                        v3.VBtn(
                            "Delete",
                            size="x-small",
                            disabled=(
                                "transfer_functions[active_transfer_function].control_points.length <= 2",
                            ),
                            click=(
                                ctrl.remove_tf_control_point,
                                "[active_transfer_function,index]",
                            ),
                        )

        v3.VBtn(
            "Add Point",
            classes="mt-2",
            size="small",
            click=(ctrl.add_tf_control_point, "[active_transfer_function]"),
        )
