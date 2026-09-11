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
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
    pointer-events: none;
}

.vtkweb-mitsuba-fps {
    position: absolute;
    top: 36px;
    right: 8px;
    z-index: 25;
    padding: 2px 6px;
    border-radius: 3px;
    background: rgba(0, 0, 0, 0.55);
    color: #fff;
    font: 12px/16px monospace;
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

    previous_slot_views = {}

    @state.change("workspace_geometry", "views")
    def _sync_slots(**_):
        sync_slot_layout()
        for slot_id, item in state.vtk_slot_layout.items():
            view_id = item and item["view_id"]
            if view_id == previous_slot_views.get(slot_id):
                continue
            previous_slot_views[slot_id] = view_id
            widget = vtk_widgets_by_slot.get(slot_id)
            if widget is not None and view_id is not None:
                widget.update()
                widget.push_camera()

    def reset_render_view(view_id: str | None = None) -> None:
        if view_id in state.views:
            ctrl.reset_camera(view_id)

    ctrl.trigger("render_view_reset")(reset_render_view)

    client.Script(
        r"""
(() => {
    if (window.__vtkwebMitsubaFrameStreamInitialized) return;
    window.__vtkwebMitsubaFrameStreamInitialized = true;

    window.__vtkwebMitsubaFrameStats = new Map();
    window.__vtkwebMitsubaLatestFrame = new Map();

    const updateMitsubaFps = () => {
        const now = performance.now();
        for (const [viewId, stats] of window.__vtkwebMitsubaFrameStats.entries()) {
            const elapsed = Math.max(1, now - stats.lastSampleTime);
            const fps = (stats.framesSinceSample * 1000.0) / elapsed;
            stats.framesSinceSample = 0;
            stats.lastSampleTime = now;
            const label = document.getElementById('vtkweb-mitsuba-fps-' + viewId);
            if (label) label.textContent = fps.toFixed(1) + ' fps';
        }
    };
    window.__vtkwebMitsubaFpsTimer = window.setInterval(updateMitsubaFps, 500);

    // Render resolution is transient transport/control data, not application
    // state. Observe the actual CSS viewport on the client and report settled
    // sizes over the ordinary Trame trigger channel.
    window.__vtkwebMitsubaResizeTimers = new Map();
    window.__vtkwebMitsubaObservedElements = new WeakSet();

    const reportMitsubaSize = (element) => {
        const canvas = element.querySelector('canvas[id^="vtkweb-mitsuba-canvas-"]');
        if (!canvas) return;
        const prefix = 'vtkweb-mitsuba-canvas-';
        const viewId = canvas.id.substring(prefix.length);
        if (!viewId) return;

        const rect = element.getBoundingClientRect();
        const width = Math.max(1, Math.round(rect.width));
        const height = Math.max(1, Math.round(rect.height));

        const existing = window.__vtkwebMitsubaResizeTimers.get(viewId);
        if (existing !== undefined) window.clearTimeout(existing);
        const timer = window.setTimeout(() => {
            window.__vtkwebMitsubaResizeTimers.delete(viewId);
            const sender = window.__vtkwebSendMitsubaResize;
            if (typeof sender === 'function') sender(viewId, width, height);
        }, 150);
        window.__vtkwebMitsubaResizeTimers.set(viewId, timer);
    };

    const mitsubaResizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) reportMitsubaSize(entry.target);
    });
    window.__vtkwebMitsubaResizeObserver = mitsubaResizeObserver;

    const observeMitsubaViews = () => {
        const elements = document.querySelectorAll('.vtkweb-mitsuba-view');
        for (const element of elements) {
            if (window.__vtkwebMitsubaObservedElements.has(element)) continue;
            window.__vtkwebMitsubaObservedElements.add(element);
            const canvas = element.querySelector('canvas[id^="vtkweb-mitsuba-canvas-"]');
            if (canvas) {
                const viewId = canvas.id.substring('vtkweb-mitsuba-canvas-'.length);
                window.__vtkwebMitsubaLatestFrame.delete(viewId);
                window.__vtkwebMitsubaFrameStats.delete(viewId);
            }
            mitsubaResizeObserver.observe(element);
            reportMitsubaSize(element);
        }
    };

    const mitsubaDomObserver = new MutationObserver(observeMitsubaViews);
    window.__vtkwebMitsubaDomObserver = mitsubaDomObserver;
    if (document.body) {
        mitsubaDomObserver.observe(document.body, { childList: true, subtree: true });
    }
    window.requestAnimationFrame(observeMitsubaViews);

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const frameSocket = new WebSocket(wsProtocol + '//' + window.location.host + '/vtkweb/frame-stream');
    frameSocket.binaryType = 'arraybuffer';
    window.__vtkwebMitsubaFrameSocket = frameSocket;

    frameSocket.onmessage = async (event) => {
        if (!(event.data instanceof ArrayBuffer) || event.data.byteLength < 4) return;

        const bytes = new Uint8Array(event.data);
        const view = new DataView(event.data);
        const headerLength = view.getUint32(0, false);
        if (headerLength < 2 || 4 + headerLength > bytes.byteLength) return;

        let header;
        try {
            header = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + headerLength)));
        } catch (_error) {
            return;
        }

        const viewId = header.view_id;
        if (!viewId) return;

        let stats = window.__vtkwebMitsubaFrameStats.get(viewId);
        if (!stats) {
            stats = { framesSinceSample: 0, lastSampleTime: performance.now() };
            window.__vtkwebMitsubaFrameStats.set(viewId, stats);
        }
        stats.framesSinceSample += 1;

        const generation = Number(header.generation || 0);
        const sequence = Number(header.sequence || 0);
        const previous = window.__vtkwebMitsubaLatestFrame.get(viewId);
        if (previous && (
            generation < previous.generation ||
            (generation === previous.generation && sequence <= previous.sequence)
        )) return;
        window.__vtkwebMitsubaLatestFrame.set(viewId, { generation, sequence });

        const imageBytes = bytes.slice(4 + headerLength);
        const blob = new Blob([imageBytes], { type: header.mime_type || 'image/jpeg' });

        let bitmap;
        try {
            bitmap = await createImageBitmap(blob);
        } catch (_error) {
            return;
        }

        const latest = window.__vtkwebMitsubaLatestFrame.get(viewId);
        if (!latest || latest.generation !== generation || latest.sequence !== sequence) {
            bitmap.close();
            return;
        }

        const canvas = document.getElementById('vtkweb-mitsuba-canvas-' + viewId);
        if (!canvas) {
            bitmap.close();
            return;
        }

        if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
            canvas.width = bitmap.width;
            canvas.height = bitmap.height;
        }
        const context = canvas.getContext('2d', { alpha: false });
        context.drawImage(bitmap, 0, 0);
        bitmap.close();
    };
})();
        """
    )

    client.ClientTriggers(
        mounted="""
            window.__vtkwebSendMitsubaResize = (viewId, width, height) => {
                trigger('set_mitsuba_render_size', [viewId, width, height]);
            };

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

            window.__vtkwebMitsubaWheelSessions = new Map();

            window.__vtkwebStartMitsubaCameraDrag = (viewId, event) => {
                if (event.button < 0 || event.button > 2) return;

                event.preventDefault();
                event.stopPropagation();
                event.currentTarget?.focus();

                const element = event.currentTarget;
                const mode = event.button === 2
                    ? 'zoom'
                    : (event.button === 1 || (event.button === 0 && event.shiftKey) ? 'pan' : 'orbit');
                let lastX = event.clientX;
                let lastY = event.clientY;
                let pendingDx = 0;
                let pendingDy = 0;
                let animationFrame = null;

                const previousCursor = element.style.cursor;
                element.style.cursor = mode === 'pan' ? 'move' : (mode === 'zoom' ? 'ns-resize' : 'grabbing');

                const flush = () => {
                    animationFrame = null;
                    if (pendingDx === 0 && pendingDy === 0) return;
                    const dx = pendingDx;
                    const dy = pendingDy;
                    pendingDx = 0;
                    pendingDy = 0;
                    const height = Math.max(element.getBoundingClientRect().height, 1);
                    trigger('interact_mitsuba_camera', [viewId, mode, dx, dy, height]);
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
                    element.style.cursor = previousCursor;
                };

                window.addEventListener('mousemove', move);
                window.addEventListener('mouseup', upHandler);
            };

            window.__vtkwebMitsubaWheel = (viewId, event) => {
                event.preventDefault();
                event.stopPropagation();
                event.currentTarget?.focus();

                let session = window.__vtkwebMitsubaWheelSessions.get(viewId);
                if (!session) {
                    session = { pending: 0, animationFrame: null, idleTimer: null };
                    window.__vtkwebMitsubaWheelSessions.set(viewId, session);
                }

                let deltaPixels = event.deltaY;
                if (event.deltaMode === 1) deltaPixels *= 16;
                if (event.deltaMode === 2) {
                    deltaPixels *= Math.max(event.currentTarget?.clientHeight || 1, 1);
                }
                session.pending += deltaPixels;

                const flush = () => {
                    session.animationFrame = null;
                    if (session.pending === 0) return;
                    const delta = session.pending;
                    session.pending = 0;
                    const height = Math.max(event.currentTarget?.clientHeight || 1, 1);
                    // Match drag-dolly sensitivity while retaining the smoother
                    // wheel/trackpad scale from the previous implementation.
                    trigger('interact_mitsuba_camera', [viewId, 'zoom', 0, delta * 0.15, height]);
                };

                if (session.animationFrame === null) {
                    session.animationFrame = window.requestAnimationFrame(flush);
                }
                if (session.idleTimer !== null) window.clearTimeout(session.idleTimer);
                session.idleTimer = window.setTimeout(() => {
                    if (session.animationFrame !== null) {
                        window.cancelAnimationFrame(session.animationFrame);
                        session.animationFrame = null;
                    }
                    flush();
                    window.__vtkwebMitsubaWheelSessions.delete(viewId);
                }, 180);
            };
        """,
        before_unmount="""
            delete window.__vtkwebSendMitsubaResize;
            delete window.__vtkwebStartTileResize;
            delete window.__vtkwebStartMitsubaCameraDrag;
            delete window.__vtkwebMitsubaWheel;
            delete window.__vtkwebMitsubaWheelSessions;
        """,
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
                ref = f"render_view_{slot_id}"
                widget = vtk_widgets.VtkLocalView(
                    backend.get_render_window(slot_id),
                    ref=ref,
                    tabindex=0,
                    style="height:100%;width:100%;outline:none;",
                    interactor_events=("['EndInteraction']",),
                    EndInteraction=(
                        ctrl.sync_vtk_camera,
                        f"[vtk_slot_layout['{slot_id}'].view_id, $refs.{ref}.getCamera()]",
                    ),
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
                    '@mousedown="window.__vtkwebStartMitsubaCameraDrag(tile.view_id, $event)"',
                    '@wheel.prevent="window.__vtkwebMitsubaWheel(tile.view_id, $event)"',
                    '@contextmenu.prevent',
                    '@keydown.space.exact.prevent="trigger(\'render_view_reset\', [tile.view_id])"',
                ],
            ):
                html.Canvas(
                    classes="vtkweb-mitsuba-image",
                    id=("'vtkweb-mitsuba-canvas-' + tile.view_id",),
                )
                html.Div(
                    "0.0 fps",
                    classes="vtkweb-mitsuba-fps",
                    id=("'vtkweb-mitsuba-fps-' + tile.view_id",),
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
                    "⇄",
                    title=("views[tile.view_id]?.type === 'vtk' ? 'Switch to Mitsuba' : 'Switch to VTK'",),
                    classes="vtkweb-tile-button",
                    v_if=("tile.view_id && ['vtk', 'mitsuba'].includes(views[tile.view_id]?.type)",),
                    click=(
                        ctrl.switch_view_type,
                        "[tile.view_id, views[tile.view_id]?.type === 'vtk' ? 'mitsuba' : 'vtk']",
                    ),
                )
                html.Button(
                    "×",
                    title="Close view",
                    classes="vtkweb-tile-button",
                    v_if=("tile.view_id",),
                    click=(ctrl.remove_view, "[tile.view_id]"),
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

    @state.change("camera_revision")
    def push_vtk_cameras(**_):
        for slot_id, widget in vtk_widgets_by_slot.items():
            if state.vtk_slot_layout.get(slot_id) is not None:
                widget.push_camera()
