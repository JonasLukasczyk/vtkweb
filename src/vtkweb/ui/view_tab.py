from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from vtkweb.rendering.base import VIEW_PROPERTY_GROUPS


def build_view_tab(
    ctrl,
) -> None:
    for group_name, group in VIEW_PROPERTY_GROUPS.items():
        with html.Details(
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
                with html.Div(
                    v_for=(
                        "property in view_property_specs.filter("
                        f"property => property.group === '{group_name}' && property.ui !== false)"
                    ),
                    key=("property.name",),
                    classes="vtkweb-view-property",
                ):
                    with html.Label(
                        v_if="property.kind === 'color'",
                        classes="vtkweb-color-box",
                    ):
                        html.Span(
                            "{{ property.label }}",
                            classes="vtkweb-control-label",
                        )
                        html.Input(
                            type="color",
                            value=("views[active_view_id]?.properties?.[property.name]",),
                            input=(
                                ctrl.set_view_property,
                                "[active_view_id,property.name,$event.target.value]",
                            ),
                        )

                    with html.Label(
                        v_if=(
                            "property.kind === 'int' || property.kind === 'float'"
                        ),
                        classes="vtkweb-input-box",
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
                            value=("views[active_view_id]?.properties?.[property.name]",),
                            change=(
                                ctrl.set_view_property,
                                "[active_view_id,property.name,Number($event.target.value)]",
                            ),
                        )

                    with html.Label(
                        v_if="property.kind === 'bool'",
                        classes="vtkweb-bool-row",
                    ):
                        html.Span(
                            "{{ property.label }}",
                            classes="vtkweb-control-label",
                        )
                        html.Input(
                            type="checkbox",
                            checked=("Boolean(views[active_view_id]?.properties?.[property.name])",),
                            change=(
                                ctrl.set_view_property,
                                "[active_view_id,property.name,$event.target.checked]",
                            ),
                        )

                    with html.Label(
                        v_if="property.kind === 'choice'",
                        classes="vtkweb-input-box",
                    ):
                        html.Span(
                            "{{ property.label }}",
                            classes="vtkweb-control-label",
                        )
                        with html.Select(
                            value=("views[active_view_id]?.properties?.[property.name]",),
                            change=(
                                ctrl.set_view_property,
                                "[active_view_id,property.name,$event.target.value]",
                            ),
                        ):
                            html.Option(
                                "{{ option.label }}",
                                v_for="option in property.options || []",
                                key=("option.value",),
                                value=("option.value",),
                            )

                    with html.Label(
                        v_if="property.kind === 'str'",
                        classes="vtkweb-input-box",
                    ):
                        html.Span(
                            "{{ property.label }}",
                            classes="vtkweb-control-label",
                        )
                        html.Input(
                            type="text",
                            value=("views[active_view_id]?.properties?.[property.name] ?? ''",),
                            change=(
                                ctrl.set_view_property,
                                "[active_view_id,property.name,$event.target.value]",
                            ),
                        )
