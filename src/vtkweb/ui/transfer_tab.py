from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def initialize_transfer_tab(state, ctrl) -> None:
    state.active_transfer_array = None

    def set_active_transfer_array(array_name: str | None) -> None:
        state.active_transfer_array = array_name

    ctrl.set_active_transfer_array = set_active_transfer_array

    @state.change("transfer_functions")
    def sync_active_transfer(**_):
        keys = list(state.transfer_functions.keys())
        if state.active_transfer_array not in keys:
            state.active_transfer_array = keys[0] if keys else None


def build_transfer_tab(ctrl) -> None:
    with html.Div(v_if=("Object.keys(transfer_functions).length === 0",)):
        html.Div(
            "No transfer functions yet. Add an array in the representations tab.",
            classes="text-medium-emphasis pa-2",
        )

    with html.Div(v_else=True):
        with html.Div(classes="vtkweb-select-box mb-2"):
            html.Span("Array", classes="vtkweb-control-label")
            v3.VSelect(
                model_value=("active_transfer_array",),
                items=("Object.keys(transfer_functions)",),
                density="compact",
                variant="plain",
                hide_details=True,
                update_modelValue=(ctrl.set_active_transfer_array, "[$event]"),
            )

        with html.Div(classes="vtkweb-select-box mb-2"):
            html.Span("Preset", classes="vtkweb-control-label")
            v3.VSelect(
                items=("tf_preset_items",),
                item_title="title",
                item_value="value",
                density="compact",
                variant="plain",
                hide_details=True,
                model_value=("null",),
                update_modelValue=(
                    ctrl.apply_tf_preset,
                    "[active_transfer_array,$event]",
                ),
            )

        with html.Div(classes="vtkweb-range-row mb-2"):
            html.Input(
                type="number",
                step="any",
                value=("transfer_functions[active_transfer_array]?.range?.[0] ?? 0",),
                classes="vtkweb-range-input",
                change=(
                    ctrl.set_tf_range,
                    "[active_transfer_array,Number($event.target.value),transfer_functions[active_transfer_array].range[1]]",
                ),
            )
            html.Input(
                type="number",
                step="any",
                value=("transfer_functions[active_transfer_array]?.range?.[1] ?? 1",),
                classes="vtkweb-range-input",
                change=(
                    ctrl.set_tf_range,
                    "[active_transfer_array,transfer_functions[active_transfer_array].range[0],Number($event.target.value)]",
                ),
            )

        with html.Table(style="width:100%;font-size:12px;border-collapse:collapse;"):
            with html.Thead():
                with html.Tr():
                    for label in ("t", "R", "G", "B", "O", ""):
                        html.Th(label, style="padding:3px;text-align:center;")
            with html.Tbody():
                with html.Tr(
                    v_for=(
                        "(point, pointIndex) in transfer_functions[active_transfer_array]?.control_points || []"
                    ),
                    key=("pointIndex",),
                ):
                    for component in range(4):
                        with html.Td(style="padding:2px;"):
                            html.Input(
                                type="number",
                                min="0",
                                max="1",
                                step="0.01",
                                value=(f"point[{component}]",),
                                style="width:58px;",
                                change=(
                                    ctrl.set_tf_control_point,
                                    f"[active_transfer_array,pointIndex,{component},Number($event.target.value)]",
                                ),
                            )
                    with html.Td(style="padding:2px;text-align:center;"):
                        html.Span("1.0")
                    with html.Td(style="padding:2px;text-align:center;"):
                        v3.VBtn(
                            icon="mdi-delete",
                            size="x-small",
                            variant="text",
                            disabled=(
                                "transfer_functions[active_transfer_array].control_points.length <= 2",
                            ),
                            click=(
                                ctrl.remove_tf_control_point,
                                "[active_transfer_array,pointIndex]",
                            ),
                        )

        with html.Div(classes="d-flex ga-2 mt-2"):
            v3.VBtn(
                "Add point",
                size="small",
                prepend_icon="mdi-plus",
                click=(ctrl.add_tf_control_point, "[active_transfer_array]"),
            )
            v3.VBtn(
                "Delete TF",
                size="small",
                variant="text",
                prepend_icon="mdi-delete",
                click=(ctrl.delete_transfer_function, "[active_transfer_array]"),
            )
