from __future__ import annotations

from trame.widgets import html


def build_view_tab(
    ctrl,
) -> None:
    with html.Div(
        v_for=("property in Object.values(views[active_view_id]?.properties || {})"),
        key=("property.name",),
        classes="vtkweb-view-property",
    ):
        with html.Label(
            v_if="property.ui !== false && property.kind === 'color'",
            classes="vtkweb-color-box mt-1",
        ):
            html.Span(
                "{{ property.label }}",
                classes="vtkweb-control-label",
            )
            html.Input(
                type="color",
                value=("property.value",),
                input=(
                    ctrl.set_view_property,
                    "[active_view_id,property.name,$event.target.value]",
                ),
            )

        with html.Label(
            v_if=(
                "property.ui !== false && "
                "(property.kind === 'int' || property.kind === 'float')"
            ),
            classes="vtkweb-input-box mt-1",
        ):
            html.Span(
                "{{ property.label }}",
                classes="vtkweb-control-label",
            )
            html.Input(
                type="number",
                min=("property.min ?? null",),
                max=("property.max ?? null",),
                step=("property.step ?? (property.kind === 'int' ? 1 : 'any')",),
                value=("property.value",),
                change=(
                    ctrl.set_view_property,
                    "[active_view_id,property.name,Number($event.target.value)]",
                ),
            )

        with html.Label(
            v_if="property.ui !== false && property.kind === 'bool'",
            classes="vtkweb-bool-row mt-1",
        ):
            html.Span(
                "{{ property.label }}",
                classes="vtkweb-control-label",
            )
            html.Input(
                type="checkbox",
                checked=("Boolean(property.value)",),
                change=(
                    ctrl.set_view_property,
                    "[active_view_id,property.name,$event.target.checked]",
                ),
            )

        with html.Label(
            v_if="property.ui !== false && property.kind === 'str'",
            classes="vtkweb-input-box mt-1",
        ):
            html.Span(
                "{{ property.label }}",
                classes="vtkweb-control-label",
            )
            html.Input(
                type="text",
                value=("property.value ?? ''",),
                change=(
                    ctrl.set_view_property,
                    "[active_view_id,property.name,$event.target.value]",
                ),
            )
