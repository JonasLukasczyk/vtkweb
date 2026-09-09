from __future__ import annotations

from trame.widgets import client, html
from trame.widgets import vtk as vtk_widgets

from vtkweb.rendering import RenderManager


WORKSPACE_STYLE = """
.vtkweb-workspace {
    position: relative;
    width: 100%;
    height: 100%;
    min-width: 0;
    min-height: 0;
    overflow: hidden;
    background: #111;
}

.vtkweb-workspace-tile,
.vtkweb-vtk-slot,
.vtkweb-mitsuba-view {
    position: absolute;
    box-sizing: border-box;
    overflow: hidden;
    border-radius: 6px;
}

.vtkweb-workspace-tile {
    pointer-events: none;
    border: 1px solid rgba(128, 128, 128, 0.2);
    z-index: 20;
}

.vtkweb-workspace-tile-active {
    border: 2px solid #2196f3;
    border-radius: 6px;
    box-shadow: inset 0 0 0 1px rgba(33, 150, 243, 0.25);
}

.vtkweb-tile-toolbar {
    position: absolute;
    top: 6px;
    right: 6px;
    display: flex;
    gap: 3px;
    z-index: 30;
    pointer-events: auto;
}

.vtkweb-tile-button {
    min-width: 24px;
    height: 24px;
    padding: 0 5px;
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 3px;
    background: rgba(20, 20, 20, 0.75);
    color: #ddd;
    cursor: pointer;
    font: 11px sans-serif;
}

.vtkweb-dummy-view,
.vtkweb-empty-view {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 6px;
    font-family: sans-serif;
}

.vtkweb-dummy-view {
    color: #ddd;
    background: repeating-linear-gradient(
        135deg,
        #202020,
        #202020 10px,
        #252525 10px,
        #252525 20px
    );
}

.vtkweb-empty-view {
    color: #777;
    background: #181818;
}

.vtkweb-view-chooser {
    pointer-events: auto;
}

.vtkweb-view-chooser-title {
    color: #aaa;
    margin-bottom: 6px;
}

.vtkweb-view-chooser-buttons {
    display: flex;
    gap: 8px;
}

.vtkweb-view-choice-button {
    min-width: 92px;
    padding: 8px 12px;
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 4px;
    color: #ddd;
    background: #252525;
    cursor: pointer;
}

.vtkweb-view-choice-button:hover {
    background: #303030;
}

.vtkweb-mitsuba-view {
    inset: 0;
    z-index: 5;
    background: #1a1a1a;
    pointer-events: auto;
    outline: none;
    cursor: grab;
    user-select: none;
}

.vtkweb-mitsuba-view:active {
    cursor: grabbing;
}

.vtkweb-mitsuba-image {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    pointer-events: none;
}

.vtkweb-tile-splitter {
    position: absolute;
    z-index: 40;
    background: transparent;
}

.vtkweb-tile-splitter.vertical {
    width: 7px;
    margin-left: -3.5px;
    cursor: col-resize;
}

.vtkweb-tile-splitter.horizontal {
    height: 7px;
    margin-top: -3.5px;
    cursor: row-resize;
}

.vtkweb-tile-splitter::after {
    content: "";
    position: absolute;
    background: rgba(160, 160, 160, 0.35);
}

.vtkweb-tile-splitter.vertical::after {
    left: 3px;
    top: 0;
    bottom: 0;
    width: 1px;
}

.vtkweb-tile-splitter.horizontal::after {
    top: 3px;
    left: 0;
    right: 0;
    height: 1px;
}

.vtkweb-tile-splitter:hover::after {
    background: rgba(210, 210, 210, 0.8);
}
"""


def build_render_view(
    state,
    ctrl,
    rendering: RenderManager,
) -> None:
    """Build the heterogeneous tiled rendering workspace."""

    backend = rendering.backend
    client.Style(WORKSPACE_STYLE)

    vtk_widgets_by_slot = {}

    def set_mitsuba_camera(view_id: str, camera: dict) -> None:
        rendering.set_mitsuba_camera_state(view_id, camera)

    ctrl.trigger("set_mitsuba_camera")(set_mitsuba_camera)

    def sync_slot_layout(**_):
        layout = {slot_id: None for slot_id in rendering.backend_slots}
        tiles_by_view = {
            tile.get("view_id"): tile
            for tile in state.workspace_geometry.get("tiles", [])
            if tile.get("view_id") is not None
        }
        for view_id, value in state.views.items():
            if value.get("type") != "vtk":
                continue
            tile = tiles_by_view.get(view_id)
            if tile is None:
                continue
            slot_id = value["backend_id"]
            layout[slot_id] = {
                "view_id": view_id,
                "container_id": tile["container_id"],
                "style": tile["style"],
            }
        state.vtk_slot_layout = layout

    sync_slot_layout()

    @state.change("workspace_geometry", "views")
    def _sync_slots(**_):
        sync_slot_layout()

    def reset_render_view(view_id: str | None = None) -> None:
        if view_id is None or view_id not in state.views:
            return
        value = state.views[view_id]
        view_type = value.get("type")
        if view_type == "mitsuba":
            ctrl.reset_camera(view_id)
            return

        if view_type == "vtk":
            # Keyboard camera reset belongs to the focused VtkLocalView.
            widget = vtk_widgets_by_slot.get(value["backend_id"])
            if widget is not None:
                widget.reset_camera()
                widget.update()

    ctrl.trigger("render_view_reset")(reset_render_view)

    client.ClientTriggers(
        mounted="""
            window.__vtkwebStartTileResize = (splitter, event) => {
                event.preventDefault();
                event.stopPropagation();

                const pane = window.document.getElementById('vtkweb-right-pane');
                if (!pane) return;

                const rect = pane.getBoundingClientRect();
                const vertical = splitter.orientation === 'vertical';
                window.document.body.style.cursor = vertical ? 'col-resize' : 'row-resize';
                window.document.body.style.userSelect = 'none';

                const move = (moveEvent) => {
                    const position = vertical
                        ? ((moveEvent.clientX - rect.left) / rect.width) * 100.0
                        : ((moveEvent.clientY - rect.top) / rect.height) * 100.0;
                    const start = vertical ? splitter.parent_left : splitter.parent_top;
                    const extent = vertical ? splitter.parent_width : splitter.parent_height;
                    const ratio = Math.max(0.1, Math.min(0.9, (position - start) / extent));
                    trigger('set_split_ratio', [splitter.id, ratio]);
                };

                const up = () => {
                    window.removeEventListener('mousemove', move);
                    window.removeEventListener('mouseup', up);
                    window.document.body.style.cursor = '';
                    window.document.body.style.userSelect = '';
                };

                window.addEventListener('mousemove', move);
                window.addEventListener('mouseup', up);
            };

            window.__vtkwebStartMitsubaOrbit = (viewId, event) => {
                if (event.button !== 0) return;

                event.preventDefault();
                event.stopPropagation();
                event.currentTarget?.focus();

                const source = views[viewId]?.camera;
                if (!source) return;

                const camera = JSON.parse(JSON.stringify(source));
                let lastX = event.clientX;
                let lastY = event.clientY;
                let pendingDx = 0;
                let pendingDy = 0;
                let animationFrame = null;

                const normalize = (v) => {
                    const n = Math.hypot(v[0], v[1], v[2]) || 1;
                    return [v[0] / n, v[1] / n, v[2] / n];
                };
                const cross = (a, b) => [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ];
                const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
                const rotate = (v, axis, angle) => {
                    axis = normalize(axis);
                    const c = Math.cos(angle);
                    const q = Math.sin(angle);
                    const axv = cross(axis, v);
                    const d = dot(axis, v) * (1 - c);
                    return [
                        v[0]*c + axv[0]*q + axis[0]*d,
                        v[1]*c + axv[1]*q + axis[1]*d,
                        v[2]*c + axv[2]*q + axis[2]*d,
                    ];
                };

                const applyOrbit = (dx, dy) => {
                    const center = camera.center_of_rotation || camera.target;
                    let offset = [
                        camera.position[0] - center[0],
                        camera.position[1] - center[1],
                        camera.position[2] - center[2],
                    ];
                    let up = normalize(camera.up);
                    const radiansPerPixel = 0.35 * Math.PI / 180.0;

                    offset = rotate(offset, up, -dx * radiansPerPixel);
                    const forward = normalize([-offset[0], -offset[1], -offset[2]]);
                    let right = cross(forward, up);
                    if (Math.hypot(...right) > 1e-12) {
                        right = normalize(right);
                        const pitch = -dy * radiansPerPixel;
                        offset = rotate(offset, right, pitch);
                        up = normalize(rotate(up, right, pitch));
                    }

                    camera.position = [
                        center[0] + offset[0],
                        center[1] + offset[1],
                        center[2] + offset[2],
                    ];
                    camera.target = [...center];
                    camera.up = up;
                    camera.center_of_rotation = [...center];
                };

                const flush = () => {
                    animationFrame = null;
                    if (pendingDx === 0 && pendingDy === 0) return;
                    const dx = pendingDx;
                    const dy = pendingDy;
                    pendingDx = 0;
                    pendingDy = 0;
                    applyOrbit(dx, dy);
                    // Send an absolute camera snapshot. Intermediate snapshots
                    // may be overwritten while the server is rendering; only
                    // the newest camera state matters.
                    trigger('set_mitsuba_camera', [viewId, camera]);
                };

                const move = (moveEvent) => {
                    pendingDx += moveEvent.clientX - lastX;
                    pendingDy += moveEvent.clientY - lastY;
                    lastX = moveEvent.clientX;
                    lastY = moveEvent.clientY;
                    if (animationFrame === null) {
                        animationFrame = window.requestAnimationFrame(flush);
                    }
                };

                const upHandler = () => {
                    window.removeEventListener('mousemove', move);
                    window.removeEventListener('mouseup', upHandler);
                    if (animationFrame !== null) {
                        window.cancelAnimationFrame(animationFrame);
                        animationFrame = null;
                    }
                    flush();
                };

                window.addEventListener('mousemove', move);
                window.addEventListener('mouseup', upHandler);
            };
        """,
        before_unmount="""
            delete window.__vtkwebStartTileResize;
            delete window.__vtkwebStartMitsubaOrbit;
        """,
    )

    # Front-end trigger used by the splitter drag handler.
    ctrl.trigger("set_split_ratio")(
        lambda container_id, ratio: ctrl.set_split_ratio(container_id, ratio)
    )

    with html.Div(classes="vtkweb-workspace"):
        # The VTK widgets are created once, one per backend slot. Their logical
        # view assignment and geometry are driven entirely by serialized state.
        for slot_id in rendering.backend_slots:
            with html.Div(
                classes="vtkweb-vtk-slot",
                v_show=(f"vtk_slot_layout['{slot_id}'] !== null",),
                style=(f"(vtk_slot_layout['{slot_id}']?.style || '') + 'z-index:10;'",),
                click=(
                    ctrl.set_active_view,
                    f"[vtk_slot_layout['{slot_id}'].view_id]",
                ),
            ):
                widget = vtk_widgets.VtkLocalView(
                    backend.get_render_window(slot_id),
                    ref=f"render_view_{slot_id}",
                    tabindex=0,
                    style="height:100%;width:100%;outline:none;",
                    focus=(
                        ctrl.set_active_view,
                        f"[vtk_slot_layout['{slot_id}'].view_id]",
                    ),
                    raw_attrs=[
                        f"@keydown.space.exact.prevent=\"trigger('render_view_reset', [vtk_slot_layout['{slot_id}'].view_id])\""
                    ],
                )
                vtk_widgets_by_slot[slot_id] = widget

        # Dummy and empty content are ordinary Vue/HTML and therefore need no
        # backend slots.
        with html.Div(
            v_for=("tile in workspace_geometry.tiles", "tile.container_id"),
            classes=(
                "['vtkweb-workspace-tile', "
                "tile.view_id === active_view_id "
                "? 'vtkweb-workspace-tile-active' : '']",
            ),
            style=("tile.style",),
        ):
            with html.Div(
                v_if=("tile.view_id && views[tile.view_id]?.type === 'dummy'",),
                classes="vtkweb-dummy-view",
            ):
                html.Div("{{ views[tile.view_id]?.name || 'Dummy view' }}")
                html.Small("{{ views[tile.view_id]?.message || 'Dummy backend' }}")

            with html.Div(
                v_if=("tile.view_id && views[tile.view_id]?.type === 'mitsuba'",),
                classes="vtkweb-mitsuba-view",
                tabindex=0,
                click=(ctrl.set_active_view, "[tile.view_id]"),
                raw_attrs=[
                    '@mousedown.left="window.__vtkwebStartMitsubaOrbit(tile.view_id, $event)"',
                    "@keydown.space.exact.prevent=\"trigger('render_view_reset', [tile.view_id])\"",
                ],
            ):
                html.Img(
                    src=("mitsuba_frames[tile.view_id] || ''",),
                    classes="vtkweb-mitsuba-image",
                    draggable="false",
                )

            with html.Div(
                v_if=("!tile.view_id",),
                classes="vtkweb-empty-view vtkweb-view-chooser",
            ):
                html.Div("Choose view", classes="vtkweb-view-chooser-title")
                with html.Div(classes="vtkweb-view-chooser-buttons"):
                    html.Button(
                        "VTK",
                        classes="vtkweb-view-choice-button",
                        click=(
                            ctrl.create_view_in_container,
                            "[tile.container_id, 'vtk']",
                        ),
                    )
                    html.Button(
                        "Mitsuba",
                        classes="vtkweb-view-choice-button",
                        click=(
                            ctrl.create_view_in_container,
                            "[tile.container_id, 'mitsuba']",
                        ),
                    )

            with html.Div(classes="vtkweb-tile-toolbar"):
                html.Button(
                    "R",
                    title="Reset camera",
                    classes="vtkweb-tile-button",
                    v_if=("tile.view_id && views[tile.view_id]?.type !== 'dummy'",),
                    click="trigger('render_view_reset', [tile.view_id])",
                )
                html.Button(
                    "V",
                    title="Split vertically",
                    classes="vtkweb-tile-button",
                    click=(
                        ctrl.split_view_container,
                        "[tile.container_id, 'vertical']",
                    ),
                )
                html.Button(
                    "H",
                    title="Split horizontally",
                    classes="vtkweb-tile-button",
                    click=(
                        ctrl.split_view_container,
                        "[tile.container_id, 'horizontal']",
                    ),
                )

        html.Div(
            v_for=("splitter in workspace_geometry.splitters", "splitter.id"),
            classes=("['vtkweb-tile-splitter', splitter.orientation]",),
            style=("splitter.style",),
            raw_attrs=['@mousedown="window.__vtkwebStartTileResize(splitter, $event)"'],
        )

    @state.change("render_revision")
    def update_render_views(**_):
        for slot_id, widget in vtk_widgets_by_slot.items():
            if state.vtk_slot_layout.get(slot_id) is not None:
                widget.update()
