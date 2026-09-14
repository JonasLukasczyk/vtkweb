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
    /* Keep geometry identical between active and inactive views. The remote
       view fills the tile's padding box, so changing border width would
       change the ResizeObserver dimensions and trigger a render resize. */
    border: 2px solid transparent;
    z-index: 20;
}

.vtkweb-workspace-tile-active {
    border-color: #2196f3;
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
    window.__vtkwebRemoteSizeRevision = new Map();

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
            if (typeof sender === 'function') {
                sender(viewId, width, height);
            }
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
                for (const key of window.__vtkwebRemoteLatestFrame.keys()) {
                    if (key === viewId || key.startsWith(viewId + ':')) {
                        window.__vtkwebRemoteLatestFrame.delete(key);
                    }
                }
                window.__vtkwebRemoteFrameStats.delete(viewId);
                window.__vtkwebRemoteSizeRevision.delete(viewId);
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

    const sendFrameCredit = () => {
        if (frameSocket.readyState === WebSocket.OPEN) {
            frameSocket.send(JSON.stringify({ type: 'ready' }));
        }
    };

    frameSocket.onopen = () => {
        // One outstanding credit is enough. If no frame is currently cached,
        // rank 0 keeps this credit until a tile changes.
        sendFrameCredit();
    };

    const decodeRemoteTile = async (packetBytes) => {
        if (!(packetBytes instanceof Uint8Array) || packetBytes.byteLength < 4) return null;
        const packetView = new DataView(
            packetBytes.buffer,
            packetBytes.byteOffset,
            packetBytes.byteLength,
        );
        const headerLength = packetView.getUint32(0, false);
        if (headerLength < 2 || 4 + headerLength > packetBytes.byteLength) return null;

        let header;
        try {
            header = JSON.parse(
                new TextDecoder().decode(packetBytes.subarray(4, 4 + headerLength))
            );
        } catch (_error) {
            return null;
        }

        const viewId = header.view_id;
        if (!viewId) return null;

        const generation = Number(header.generation || 0);
        const sequence = Number(header.sequence || 0);
        const sizeRevision = Number(header.size_revision || 0);
        const tileId = Number(header.tile_id || 0);
        const frameKey = viewId + ':' + tileId;
        const previous = window.__vtkwebRemoteLatestFrame.get(frameKey);
        if (previous && (
            generation < previous.generation ||
            (generation === previous.generation && sizeRevision < previous.sizeRevision) ||
            (generation === previous.generation &&
             sizeRevision === previous.sizeRevision &&
             sequence <= previous.sequence)
        )) return null;

        const imageBytes = packetBytes.slice(4 + headerLength);
        const blob = new Blob([imageBytes], { type: header.mime_type || 'image/jpeg' });

        let bitmap;
        try {
            bitmap = await createImageBitmap(blob);
        } catch (_error) {
            return null;
        }

        // Commit freshness only after the encoded image decoded successfully.
        window.__vtkwebRemoteLatestFrame.set(
            frameKey,
            { generation, sequence, sizeRevision },
        );
        return { header, bitmap, viewId, tileId };
    };

    const drawRemoteBatch = (tiles) => {
        const touchedViews = new Set();

        // A batch may contain late tiles from a previous framebuffer size. For
        // each view, choose the newest size revision represented in this batch
        // (or already displayed) and never compose older revisions with it.
        const targetRevision = new Map(window.__vtkwebRemoteSizeRevision);
        for (const tile of tiles) {
            if (!tile) continue;
            const revision = Number(tile.header.size_revision || 0);
            const previous = Number(targetRevision.get(tile.viewId) || 0);
            if (revision > previous) targetRevision.set(tile.viewId, revision);
        }

        for (const tile of tiles) {
            if (!tile) continue;
            const { header, bitmap, viewId, tileId } = tile;
            const canvas = document.getElementById('vtkweb-remote-canvas-' + viewId);
            if (!canvas) {
                bitmap.close();
                continue;
            }

            const region = Array.isArray(header.region) ? header.region : null;
            const fullSize = Array.isArray(header.full_size) ? header.full_size : null;
            const targetWidth = fullSize ? Number(fullSize[0]) : bitmap.width;
            const targetHeight = fullSize ? Number(fullSize[1]) : bitmap.height;
            const sizeRevision = Number(header.size_revision || 0);
            const newestRevision = Number(targetRevision.get(viewId) || 0);
            if (sizeRevision < newestRevision) {
                bitmap.close();
                continue;
            }

            const displayedRevision = Number(
                window.__vtkwebRemoteSizeRevision.get(viewId) || 0
            );
            if (
                sizeRevision > displayedRevision ||
                canvas.width !== targetWidth ||
                canvas.height !== targetHeight
            ) {
                canvas.width = targetWidth;
                canvas.height = targetHeight;
                window.__vtkwebRemoteSizeRevision.set(viewId, sizeRevision);
            }

            const context = canvas.getContext('2d', { alpha: false });
            const x = region ? Number(region[0]) : 0;
            const y = region ? Number(region[1]) : 0;
            context.drawImage(bitmap, x, y);

            if (header.debug && region) {
                const colors = [
                    '#ff3b30', '#34c759', '#007aff', '#ffcc00',
                    '#af52de', '#00c7be', '#ff9500', '#ff2d55',
                ];
                const lineWidth = 2;
                const inset = lineWidth / 2;
                const width = Number(region[2]);
                const height = Number(region[3]);
                context.save();
                context.strokeStyle = colors[tileId % colors.length];
                context.lineWidth = lineWidth;
                context.strokeRect(
                    x + inset, y + inset,
                    Math.max(0, width - lineWidth),
                    Math.max(0, height - lineWidth),
                );
                context.restore();
            }

            bitmap.close();
            touchedViews.add(viewId);
        }

        // FPS measures browser paint cadence rather than tile-message rate.
        const now = performance.now();
        for (const viewId of touchedViews) {
            let stats = window.__vtkwebRemoteFrameStats.get(viewId);
            if (!stats) {
                stats = { framesSinceSample: 0, lastSampleTime: now };
                window.__vtkwebRemoteFrameStats.set(viewId, stats);
            }
            stats.framesSinceSample += 1;
        }
    };

    frameSocket.onmessage = async (event) => {
        if (!(event.data instanceof ArrayBuffer) || event.data.byteLength < 8) return;

        const bytes = new Uint8Array(event.data);
        if (
            bytes[0] !== 0x56 || bytes[1] !== 0x54 ||
            bytes[2] !== 0x42 || bytes[3] !== 0x31
        ) return; // "VTB1"

        const batchView = new DataView(event.data);
        const count = batchView.getUint32(4, false);
        let offset = 8;
        const packets = [];

        for (let index = 0; index < count; index += 1) {
            if (offset + 4 > bytes.byteLength) break;
            const packetLength = batchView.getUint32(offset, false);
            offset += 4;
            if (packetLength < 4 || offset + packetLength > bytes.byteLength) break;
            packets.push(bytes.slice(offset, offset + packetLength));
            offset += packetLength;
        }

        const decoded = await Promise.all(packets.map(decodeRemoteTile));
        await new Promise((resolve) => {
            window.requestAnimationFrame(() => {
                drawRemoteBatch(decoded);
                resolve();
            });
        });

        // Return exactly one credit after this batch is painted. Rank 0 then
        // sends only cache entries updated since the batch we just consumed.
        sendFrameCredit();
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
