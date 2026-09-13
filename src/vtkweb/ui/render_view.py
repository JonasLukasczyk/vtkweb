from __future__ import annotations

from trame.widgets import client, html
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
.vtkweb-remote-view {
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

.vtkweb-remote-view {
    inset: 0;
    z-index: 5;
    background: #1a1a1a;
    pointer-events: auto;
    outline: none;
    cursor: grab;
    user-select: none;
}

.vtkweb-remote-view:active {
    cursor: grabbing;
}

.vtkweb-remote-image {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
    pointer-events: none;
}

.vtkweb-remote-fps {
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

    client.Style(WORKSPACE_STYLE)

    def reset_render_view(view_id: str | None = None) -> None:
        view = state.views.get(view_id) if view_id is not None else None
        if view is not None and view.get("type") in {"vtk", "mitsuba"}:
            ctrl.reset_camera(view_id)

    ctrl.trigger("render_view_reset")(reset_render_view)

    client.Script(
        r"""
(() => {
    if (window.__vtkwebRemoteFrameStreamInitialized) return;
    window.__vtkwebRemoteFrameStreamInitialized = true;

    window.__vtkwebRemoteFrameStats = new Map();
    window.__vtkwebRemoteLatestFrame = new Map();

    const updateRemoteFps = () => {
        const now = performance.now();
        for (const [viewId, stats] of window.__vtkwebRemoteFrameStats.entries()) {
            const elapsed = Math.max(1, now - stats.lastSampleTime);
            const fps = (stats.framesSinceSample * 1000.0) / elapsed;
            stats.framesSinceSample = 0;
            stats.lastSampleTime = now;
            const label = document.getElementById('vtkweb-remote-fps-' + viewId);
            if (label) label.textContent = fps.toFixed(1) + ' fps';
        }
    };
    window.__vtkwebRemoteFpsTimer = window.setInterval(updateRemoteFps, 500);

    // Render resolution is transient transport/control data, not application
    // state. Observe the actual CSS viewport on the client and report settled
    // sizes over the ordinary Trame trigger channel.
    window.__vtkwebRemoteResizeTimers = new Map();
    window.__vtkwebRemoteObservedElements = new WeakSet();

    const reportRemoteSize = (element) => {
        const canvas = element.querySelector('canvas[id^="vtkweb-remote-canvas-"]');
        if (!canvas) return;
        const prefix = 'vtkweb-remote-canvas-';
        const viewId = canvas.id.substring(prefix.length);
        if (!viewId) return;

        const rect = element.getBoundingClientRect();
        const width = Math.max(1, Math.round(rect.width));
        const height = Math.max(1, Math.round(rect.height));

        const existing = window.__vtkwebRemoteResizeTimers.get(viewId);
        if (existing !== undefined) window.clearTimeout(existing);
        const timer = window.setTimeout(() => {
            window.__vtkwebRemoteResizeTimers.delete(viewId);
            const sender = window.__vtkwebSendRemoteResize;
            if (typeof sender === 'function') sender(viewId, width, height);
        }, 150);
        window.__vtkwebRemoteResizeTimers.set(viewId, timer);
    };

    // Expose the same measurement path so the server can explicitly request
    // fresh viewport sizes after lifecycle operations where no DOM resize occurs.
    window.__vtkwebReportRemoteSize = reportRemoteSize;
    window.__vtkwebRequestRemoteSizes = () => {
        const elements = document.querySelectorAll('.vtkweb-remote-view');
        for (const element of elements) reportRemoteSize(element);
    };

    const remoteResizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) reportRemoteSize(entry.target);
    });
    window.__vtkwebRemoteResizeObserver = remoteResizeObserver;

    const observeRemoteViews = () => {
        const elements = document.querySelectorAll('.vtkweb-remote-view');
        for (const element of elements) {
            if (window.__vtkwebRemoteObservedElements.has(element)) continue;
            window.__vtkwebRemoteObservedElements.add(element);
            const canvas = element.querySelector('canvas[id^="vtkweb-remote-canvas-"]');
            if (canvas) {
                const viewId = canvas.id.substring('vtkweb-remote-canvas-'.length);
                window.__vtkwebRemoteLatestFrame.delete(viewId);
                window.__vtkwebRemoteFrameStats.delete(viewId);
            }
            remoteResizeObserver.observe(element);
            reportRemoteSize(element);
        }
    };

    const remoteDomObserver = new MutationObserver(observeRemoteViews);
    window.__vtkwebRemoteDomObserver = remoteDomObserver;
    if (document.body) {
        remoteDomObserver.observe(document.body, { childList: true, subtree: true });
    }
    window.requestAnimationFrame(observeRemoteViews);

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const frameSocket = new WebSocket(wsProtocol + '//' + window.location.host + '/vtkweb/frame-stream');
    frameSocket.binaryType = 'arraybuffer';
    window.__vtkwebRemoteFrameSocket = frameSocket;

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

        let stats = window.__vtkwebRemoteFrameStats.get(viewId);
        if (!stats) {
            stats = { framesSinceSample: 0, lastSampleTime: performance.now() };
            window.__vtkwebRemoteFrameStats.set(viewId, stats);
        }
        stats.framesSinceSample += 1;

        const generation = Number(header.generation || 0);
        const sequence = Number(header.sequence || 0);
        const previous = window.__vtkwebRemoteLatestFrame.get(viewId);
        if (previous && (
            generation < previous.generation ||
            (generation === previous.generation && sequence <= previous.sequence)
        )) return;
        window.__vtkwebRemoteLatestFrame.set(viewId, { generation, sequence });

        const imageBytes = bytes.slice(4 + headerLength);
        const blob = new Blob([imageBytes], { type: header.mime_type || 'image/jpeg' });

        let bitmap;
        try {
            bitmap = await createImageBitmap(blob);
        } catch (_error) {
            return;
        }

        const latest = window.__vtkwebRemoteLatestFrame.get(viewId);
        if (!latest || latest.generation !== generation || latest.sequence !== sequence) {
            bitmap.close();
            return;
        }

        const canvas = document.getElementById('vtkweb-remote-canvas-' + viewId);
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

    client.ClientStateChange(
        value="remote_render_size_request_epoch",
        change="""
            $nextTick(() => {
                window.__vtkwebRequestRemoteSizes?.();
            });
        """,
    )

    client.ClientTriggers(
        mounted="""
            window.__vtkwebSendRemoteResize = (viewId, width, height) => {
                trigger('set_render_size', [viewId, width, height]);
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

            window.__vtkwebRemoteWheelSessions = new Map();

            window.__vtkwebStartCameraDrag = (viewId, event) => {
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
                    trigger('interact_view_camera', [viewId, mode, dx, dy, height]);
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

            window.__vtkwebRemoteWheel = (viewId, event) => {
                event.preventDefault();
                event.stopPropagation();
                event.currentTarget?.focus();

                let session = window.__vtkwebRemoteWheelSessions.get(viewId);
                if (!session) {
                    session = { pending: 0, animationFrame: null, idleTimer: null };
                    window.__vtkwebRemoteWheelSessions.set(viewId, session);
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
                    trigger('interact_view_camera', [viewId, 'zoom', 0, delta * 0.15, height]);
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
                    window.__vtkwebRemoteWheelSessions.delete(viewId);
                }, 180);
            };
        """,
        before_unmount="""
            delete window.__vtkwebSendRemoteResize;
            delete window.__vtkwebStartTileResize;
            delete window.__vtkwebStartCameraDrag;
            delete window.__vtkwebRemoteWheel;
            delete window.__vtkwebRemoteWheelSessions;
        """,
    )

    with html.Div(classes="vtkweb-workspace"):
        # Render views are backend-agnostic canvases; dummy/empty views remain HTML.
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
                v_if=(
                    "tile.view_id && ['vtk', 'mitsuba'].includes(views[tile.view_id]?.type)",
                ),
                classes="vtkweb-remote-view",
                tabindex=0,
                click=(ctrl.set_active_view, "[tile.view_id]"),
                raw_attrs=[
                    '@mousedown="window.__vtkwebStartCameraDrag(tile.view_id, $event)"',
                    '@wheel.prevent="window.__vtkwebRemoteWheel(tile.view_id, $event)"',
                    "@contextmenu.prevent",
                    "@keydown.space.exact.prevent=\"trigger('render_view_reset', [tile.view_id])\"",
                ],
            ):
                html.Canvas(
                    classes="vtkweb-remote-image",
                    id=("'vtkweb-remote-canvas-' + tile.view_id",),
                )
                html.Div(
                    "0.0 fps",
                    classes="vtkweb-remote-fps",
                    id=("'vtkweb-remote-fps-' + tile.view_id",),
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
                    title=(
                        "views[tile.view_id]?.type === 'vtk' ? 'Switch to Mitsuba' : 'Switch to VTK'",
                    ),
                    classes="vtkweb-tile-button",
                    v_if=(
                        "tile.view_id && ['vtk', 'mitsuba'].includes(views[tile.view_id]?.type)",
                    ),
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
