"""Shared compact property controls for inspector tabs.

Values stay in Trame state; these builders only emit UI bindings.
"""

from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def vector_control(
    ctrl,
    *,
    value: str,
    context: str,
    update,
    step: str = "'any'",
    titles: str = "null",
    fallback: str | None = None,
) -> None:
    """Render any-length numeric vector in one row (including six components)."""
    source = f"({value} || {fallback})" if fallback else value
    with html.Div(v_if="property.kind === 'vector'", classes="vtkweb-vector-box"):
        html.Span("{{ property.label }}", classes="vtkweb-control-label")
        with html.Div(
            classes="vtkweb-vector-fields",
            style=(
                f"'grid-template-columns:repeat(' + ({source}).length + ',minmax(0,1fr))'",
            ),
        ):
            html.Input(
                v_for=f"(component,index) in {source}",
                key=("index",),
                type="number",
                step=(step,),
                title=(titles,),
                value=("component",),
                change=(
                    update,
                    f"[{context},property.name,{source}.map((v,i) => i === index ? ($event.target.value === '' ? v : Number($event.target.value)) : v)]",
                ),
            )


def boolean_control(ctrl, *, value: str, context: str, update) -> None:
    with html.Label(v_if="property.kind === 'bool'", classes="vtkweb-bool-row"):
        html.Span("{{ property.label }}", classes="vtkweb-control-label")
        html.Input(
            type="checkbox",
            checked=(f"Boolean({value})",),
            change=(update, f"[{context},property.name,$event.target.checked]"),
        )


def file_picker_button(ctrl, *, condition: str, context: str, open_dialog) -> None:
    """Use the application's existing shared file-browser dialog."""
    with v3.VBtn(
        v_if=condition,
        icon=True,
        size="x-small",
        variant="text",
        classes="vtkweb-file-picker-button",
        title="Browse server filesystem",
        click=(open_dialog, f"[{context},property.name]"),
    ):
        v3.VIcon("mdi-folder-open-outline", size="small")
