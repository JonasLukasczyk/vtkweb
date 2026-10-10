from __future__ import annotations

from trame.widgets import client, html
from trame.widgets import vuetify3 as v3
from trame_client.widgets.core import HtmlElement

from vtkweb.transfer_functions import sync_transfer_function_ui_state


class _SvgTag(HtmlElement):
    def __init__(self, tag, children=None, **kwargs):
        super().__init__(tag, children, **kwargs)
        self._attr_names += [
            ["classes", "class"],
            ["key", ":key"],
            ["v_for", "v-for"],
            "x", "y", "x1", "y1", "x2", "y2",
            "width", "height", "points", "cx", "cy", "r",
        ]


class _NativeVueTag(HtmlElement):
    """Native HTML with explicit Vue bindings for the custom preset menu."""
    def __init__(self, tag, children=None, **kwargs):
        super().__init__(tag, children, **kwargs)
        self._attr_names += [
            ["classes", "class"], ["v_for", "v-for"],
            ["v_if", "v-if"], ["key", ":key"],
            ["vue_style", ":style"], ["vue_click", "@click"],
        ]


def initialize_transfer_tab(state, ctrl) -> None:
    state.active_transfer_function = None
    state.active_tf_preset = None
    state.transfer_function_items = []

    def update_items(**_):
        sync_transfer_function_ui_state(state)

    @state.change("transfer_functions")
    def on_transfer_functions_change(**_):
        update_items()

    def set_active_transfer_function(array_name: str | None) -> None:
        state.active_transfer_function = array_name
        state.active_tf_preset = None

    ctrl.set_active_transfer_function = set_active_transfer_function
    # The native Vue menu uses a named Trame trigger. Register the existing
    # MPI-aware controller operation directly; do not introduce a UI adapter.
    ctrl.trigger("apply_tf_preset")(ctrl.apply_tf_preset)
    for name in (
        "set_tf_color_control_point_component",
        "add_tf_color_control_point",
        "remove_tf_color_control_point",
        "set_tf_color_point_rgb_at_x",
        "remove_tf_color_point_at_x",
    ):
        ctrl.trigger(name)(getattr(ctrl, name))
    update_items()


def _mapping_range_row(ctrl, mapping_name: str) -> None:
    with html.Div(classes="vtkweb-tf-range-row"):
        html.Span(mapping_name.capitalize(), classes="vtkweb-tf-range-label")
        for bound, other in ((0, 1), (1, 0)):
            html.Input(
                type="number", step="any",
                value=(f"transfer_functions[active_transfer_function]?.{mapping_name}?.range?.[{bound}] ?? {bound}",),
                classes="vtkweb-range-input",
                change=(
                    ctrl.set_tf_mapping_range,
                    f"[active_transfer_function,'{mapping_name}',"
                    + ("Number($event.target.value)," if bound == 0 else f"transfer_functions[active_transfer_function].{mapping_name}.range[0],")
                    + (f"transfer_functions[active_transfer_function].{mapping_name}.range[1]]" if bound == 0 else "Number($event.target.value)]"),
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

            // Only a stationary pointer gesture on the empty bar adds a point.
            // A handle drag can never accidentally add one.
            window.__vtkwebColorBarPointer = (arrayName, event) => {
                const bar = event.currentTarget;
                if (event.type === 'pointerdown') {
                    bar.__tfAdd = event.button === 0 && event.target.classList.contains('vtkweb-tf-color-fill')
                        ? [event.pointerId, event.clientX, event.clientY] : null;
                    return;
                }
                const start = bar.__tfAdd;
                bar.__tfAdd = null;
                if (event.type !== 'pointerup' || !arrayName || !start ||
                    !event.target.classList.contains('vtkweb-tf-color-fill') ||
                    event.pointerId !== start[0] ||
                    Math.hypot(event.clientX - start[1], event.clientY - start[2]) > 3) return;
                trigger('add_tf_color_control_point', [arrayName]);
            };
            window.__vtkwebPendingColorClick = null;
            // Server state is the only source for handle positions and gradient stops.
            window.__vtkwebColorDrag = (arrayName, index, pointX, rgb, event) => {
                if (!arrayName || event.button !== 0) return;
                event.preventDefault();
                event.stopPropagation();
                const handle = event.currentTarget;
                const bar = handle && handle.parentElement;
                if (!bar) return;
                const identityX = Number(pointX);
                const pointerId = event.pointerId;
                const startX = event.clientX;
                const startY = event.clientY;
                const rect = bar.getBoundingClientRect();
                let moved = false;
                let lastSent = identityX;
                const cleanup = () => {
                    window.removeEventListener('pointermove', move);
                    window.removeEventListener('pointerup', up);
                    window.removeEventListener('pointercancel', cancel);
                };
                const move = e => {
                    if (e.pointerId !== pointerId) return;
                    if (!moved && Math.hypot(e.clientX - startX, e.clientY - startY) < 3) return;
                    moved = true;
                    // The backend constrains interior points between neighbors,
                    // keeping the index stable exactly as in the opacity editor.
                    const x = Math.max(0, Math.min(1,
                        (e.clientX - rect.left) / Math.max(1, rect.width)));
                    if (Math.abs(x - lastSent) < 1e-5) return;
                    lastSent = x;
                    trigger('set_tf_color_control_point_component', [arrayName, index, 0, x]);
                };
                const up = e => {
                    if (e.pointerId !== pointerId) return;
                    cleanup();
                    if (!moved && handle.isConnected) {
                        if (window.__vtkwebPendingColorClick)
                            window.clearTimeout(window.__vtkwebPendingColorClick);
                        window.__vtkwebPendingColorClick = window.setTimeout(() => {
                            window.__vtkwebPendingColorClick = null;
                            if (handle.isConnected)
                                window.__vtkwebColorPicker(arrayName, identityX, rgb, handle);
                        }, 300);
                    }
                };
                const cancel = e => {
                    if (e.pointerId === pointerId) cleanup();
                };
                window.addEventListener('pointermove', move);
                window.addEventListener('pointerup', up);
                window.addEventListener('pointercancel', cancel);
            };
            window.__vtkwebColorPicker = (arrayName, identityX, rgb, handle) => {
                const hex = '#' + rgb.map(n => Math.round(Math.max(0, Math.min(1, Number(n))) * 255)
                    .toString(16).padStart(2, '0')).join('');
                const doc = handle && handle.ownerDocument;
                if (!doc) return;
                const input = doc.createElement('input');
                input.type = 'color';
                input.value = hex;
                input.style.cssText = 'position:fixed;opacity:0;pointer-events:none;width:1px;height:1px';
                doc.body.appendChild(input);
                input.addEventListener('input', () => {
                    const value = input.value;
                    const rgb = [0, 1, 2].map(i => parseInt(value.slice(1 + i * 2, 3 + i * 2), 16) / 255);
                    trigger('set_tf_color_point_rgb_at_x', [arrayName, identityX, rgb]);
                });
                input.addEventListener('change', () => input.remove(), {once:true});
                input.addEventListener('blur', () => setTimeout(() => input.remove(), 250), {once:true});
                input.click();
            };
            window.__vtkwebColorRemove = (arrayName, identityX, event) => {
                event.preventDefault();
                event.stopPropagation();
                if (window.__vtkwebPendingColorClick) {
                    window.clearTimeout(window.__vtkwebPendingColorClick);
                    window.__vtkwebPendingColorClick = null;
                }
                trigger('remove_tf_color_point_at_x', [arrayName, identityX]);
            };

            window.__vtkwebStartOpacityDrag = (arrayName, pointIndex, event) => {
                if (!arrayName || event.button !== 0) return;
                event.preventDefault();
                event.stopPropagation();
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
                };

                window.addEventListener('pointermove', move);
                window.addEventListener('pointerup', up);
                window.addEventListener('pointercancel', up);
            };
        """,
        before_unmount=r"""
            delete window.__vtkwebColorBarPointer;
            if (window.__vtkwebPendingColorClick)
                window.clearTimeout(window.__vtkwebPendingColorClick);
            delete window.__vtkwebPendingColorClick;
            delete window.__vtkwebColorDrag;
            delete window.__vtkwebColorPicker;
            delete window.__vtkwebColorRemove;
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

        # Compact toolbar: four icon-only actions, with native title tooltips.
        with html.Div(classes="vtkweb-tf-toolbar"):
            with _NativeVueTag("details", classes="vtkweb-tf-tool-menu vtkweb-tf-preset-menu"):
                with _NativeVueTag("summary", classes="vtkweb-tf-tool-button", title="Choose preset color map"):
                    v3.VIcon("mdi-palette", size="small")
                with html.Div(classes="vtkweb-colormap-dropdown-menu"):
                    with _NativeVueTag(
                        "button", v_for="item in tf_preset_items", key="item.value",
                        classes="vtkweb-colormap-dropdown-option",
                        vue_style="{ backgroundImage: 'url(' + item.preview + ')' }",
                        vue_click=(
                            "active_tf_preset = item.value; "
                            "trigger('apply_tf_preset', [active_transfer_function, item.value]); "
                            "$event.currentTarget.closest('details').open = false"
                        ),
                        type="button",
                    ):
                        html.Span("{{ item.title }}", classes="vtkweb-colormap-dropdown-option-label")

            with html.Button(
                classes="vtkweb-tf-tool-button", title="Invert color map",
                click=(ctrl.invert_tf_color_map, "[active_transfer_function]"),
                type="button",
            ):
                v3.VIcon("mdi-swap-horizontal", size="small")

            with html.Button(
                classes="vtkweb-tf-tool-button", title="Automatically adjust color and opacity ranges to data",
                click=(ctrl.rescale_tf_both_mappings, "[active_transfer_function]"),
                type="button",
            ):
                v3.VIcon("mdi-auto-fix", size="small")

            with _NativeVueTag("details", classes="vtkweb-tf-tool-menu"):
                with _NativeVueTag("summary", classes="vtkweb-tf-tool-button", title="Set data range"):
                    v3.VIcon("mdi-arrow-expand-horizontal", size="small")
                with html.Div(classes="vtkweb-tf-tool-panel"):
                    _mapping_range_row(ctrl, "color")
                    _mapping_range_row(ctrl, "opacity")

        with html.Svg(
            classes="vtkweb-opacity-editor",
            raw_attrs=[
                'viewBox="0 0 300 140"',
                'preserveAspectRatio="none"',
                '@click="window.__vtkwebAddOpacityPoint(active_transfer_function, $event)"',
            ],
        ):
            _SvgTag("rect", x="0", y="0", width="300", height="140", classes="vtkweb-opacity-bg")
            _SvgTag("line", x1="0", y1="70", x2="300", y2="70", classes="vtkweb-opacity-grid")
            _SvgTag("line", x1="150", y1="0", x2="150", y2="140", classes="vtkweb-opacity-grid")
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
                    '@click.stop',
                    '@dblclick.stop="trigger(\'remove_tf_opacity_control_point\', [active_transfer_function,index])"',
                ],
            )

        # Single reactive source for BOTH gradient stops and handles. No
        # Python-derived CSS styles, duplicate preview state, or DOM overrides.
        with html.Svg(
            classes="vtkweb-tf-color-svg",
            raw_attrs=[
                'viewBox="0 0 300 30"',
                'preserveAspectRatio="none"',
                '@pointerdown="window.__vtkwebColorBarPointer(active_transfer_function, $event)"',
                '@pointerup="window.__vtkwebColorBarPointer(active_transfer_function, $event)"',
                '@pointercancel="window.__vtkwebColorBarPointer(active_transfer_function, $event)"',
            ],
        ):
            with _SvgTag("defs"):
                with _SvgTag("linearGradient", raw_attrs=[
                    'id="vtkweb-tf-color-gradient"',
                    'x1="0%"', 'y1="0%"', 'x2="100%"', 'y2="0%"',
                ]):
                    _SvgTag(
                        "stop",
                        v_for="(point,index) in (transfer_functions[active_transfer_function]?.color?.control_points || [])",
                        key="index",
                        raw_attrs=[
                            ":offset=\"(point[0] * 100) + '%'\"",
                            ":stop-color=\"'rgb(' + point.slice(1,4).map(v => Math.round(v * 255)).join(',') + ')'\"",
                        ],
                    )
            _SvgTag("rect", x="0", y="0", width="300", height="30",
                    classes="vtkweb-tf-color-fill",
                    raw_attrs=['fill="url(#vtkweb-tf-color-gradient)"'])
            _SvgTag(
                "circle",
                v_for="(point,index) in (transfer_functions[active_transfer_function]?.color?.control_points || [])",
                key="index",
                cx=("point[0] * 300",), cy="15", r="7",
                classes="vtkweb-tf-color-svg-handle",
                raw_attrs=[
                    ":fill=\"'rgb(' + point.slice(1,4).map(v => Math.round(v * 255)).join(',') + ')'\"",
                    '@pointerdown.stop="window.__vtkwebColorDrag(active_transfer_function, index, point[0], point.slice(1,4), $event)"',
                    '@dblclick.stop="window.__vtkwebColorRemove(active_transfer_function, point[0], $event)"',
                ],
            )
