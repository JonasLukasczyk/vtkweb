from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def build_representations_tab(
    ctrl,
) -> None:
    html.Div(
        "Representations",
        classes="vtkweb-section-title",
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
                classes=(
                    "vtkweb-representation-header"
                ),
            ):
                html.Span(
                    "{{ representation.label }}",
                    classes=(
                        "vtkweb-representation-title"
                    ),
                )

                html.Button(
                    "×",
                    type="button",
                    title="Remove representation",
                    classes=(
                        "vtkweb-representation-remove"
                    ),
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
                    classes="vtkweb-select-prefix",
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
                    classes=(
                        "vtkweb-select-control"
                    ),
                    update_modelValue=(
                        ctrl.set_representation_kind,
                        (
                            "[representation.id,"
                            "$event]"
                        ),
                    ),
                )

            # Visibility
            with html.Label(
                classes="vtkweb-bool-row mt-1",
            ):
                html.Span(
                    "Visible"
                )

                html.Input(
                    type="checkbox",
                    checked=(
                        "representation.visible",
                    ),
                    change=(
                        ctrl.set_representation_visibility,
                        (
                            "[representation.id,"
                            "$event.target.checked]"
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
                        classes="vtkweb-select-prefix",
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
                        classes=(
                            "vtkweb-select-control"
                        ),
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
                        classes=(
                            "vtkweb-range-input"
                        ),
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
                        classes=(
                            "vtkweb-range-input"
                        ),
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
