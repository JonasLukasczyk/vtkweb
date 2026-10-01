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

.vtkweb-remote-frame {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
    pointer-events: none;
}

.vtkweb-render-stats {
    position: absolute;
    top: 8px;
    left: 8px;
    z-index: 25;
    padding: 5px 7px;
    border-radius: 4px;
    background: rgba(0, 0, 0, 0.45);
    color: rgba(255, 255, 255, 0.92);
    font: 11px/1.35 monospace;
    white-space: nowrap;
    pointer-events: none;
}

.vtkweb-render-stats-ranks,
.vtkweb-client-video-stats {
    margin-top: 2px;
}

.vtkweb-server-render-stats,
.vtkweb-render-stats-ranks,
.vtkweb-client-video-stats {
    white-space: pre-line;
}


.vtkweb-render-activities {
    position: absolute;
    right: 10px;
    bottom: 10px;
    z-index: 30;
    width: min(360px, calc(100% - 20px));
    display: flex;
    flex-direction: column;
    gap: 6px;
    pointer-events: none;
}

.vtkweb-render-activity {
    padding: 8px 10px;
    border-radius: 6px;
    background: rgba(0, 0, 0, 0.62);
    color: rgba(255, 255, 255, 0.95);
    font: 11px/1.35 monospace;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
}

.vtkweb-render-activity-title {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 4px;
}

.vtkweb-render-activity-details {
    opacity: 0.72;
    margin-bottom: 4px;
}

.vtkweb-render-activity-progress {
    height: 5px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.16);
}

.vtkweb-render-activity-progress > div {
    height: 100%;
    min-width: 2px;
    background: rgba(120, 190, 255, 0.78);
    transition: width 120ms linear;
}

.vtkweb-render-activity.complete {
    overflow: hidden;
    animation: vtkweb-activity-complete-hide 220ms ease 2.8s forwards;
}

.vtkweb-render-activity.complete .vtkweb-render-activity-progress > div {
    background: rgba(120, 220, 155, 0.98);
}

.vtkweb-render-activity.indeterminate .vtkweb-render-activity-progress > div {
    width: 38% !important;
    animation: vtkweb-activity-slide 1s ease-in-out infinite alternate;
}

.vtkweb-render-activity-values {
    margin-top: 4px;
    opacity: 0.9;
}

@keyframes vtkweb-activity-slide {
    from { transform: translateX(-20%); }
    to { transform: translateX(180%); }
}

@keyframes vtkweb-activity-complete-hide {
    from {
        opacity: 1;
        transform: translateY(0);
        max-height: 160px;
        padding-top: 8px;
        padding-bottom: 8px;
    }
    to {
        opacity: 0;
        transform: translateY(4px);
        max-height: 0;
        padding-top: 0;
        padding-bottom: 0;
    }
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
    if (window.__vtkwebH264Initialized) return;
    window.__vtkwebH264Initialized = true;

    const streams = new Map();
    const observed = new WeakSet();
    const resizeTimers = new Map();

    const viewIdFromCanvas = (canvas) => {
        const prefix = 'vtkweb-remote-frame-';
        return canvas?.id?.startsWith(prefix) ? canvas.id.substring(prefix.length) : null;
    };

    const closeStream = (viewId) => {
        const stream = streams.get(viewId);
        if (!stream) return;
        streams.delete(viewId);
        const resizeTimer = resizeTimers.get(viewId);
        if (resizeTimer !== undefined) {
            window.clearTimeout(resizeTimer);
            resizeTimers.delete(viewId);
        }
        stream.socket.onclose = null;
        stream.socket.close();
        if (stream.decoder.state !== 'closed') stream.decoder.close();
    };

    const connectCanvas = (canvas) => {
        const viewId = viewIdFromCanvas(canvas);
        if (!viewId || streams.has(viewId)) return;
        if (!('VideoDecoder' in window)) {
            console.error('vtkweb requires the WebCodecs VideoDecoder API');
            return;
        }

        const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${protocol}//${window.location.host}/vtkweb/video/${encodeURIComponent(viewId)}`;
        const socket = new WebSocket(url);
        socket.binaryType = 'arraybuffer';

        const clientStats = {
            startedAt: performance.now(),
            received: 0,
            decoded: 0,
            displayed: 0,
            queueSize: 0,
        };

        const publishClientStats = () => {
            const now = performance.now();
            const elapsed = (now - clientStats.startedAt) / 1000.0;
            if (elapsed < 1.0) return;
            const element = document.getElementById(`vtkweb-client-video-stats-${viewId}`);
            if (element) {
                element.textContent =
                    `Video recv: ${(clientStats.received / elapsed).toFixed(1)} fps\n` +
                    `Decoded: ${(clientStats.decoded / elapsed).toFixed(1)} fps\n` +
                    `Displayed: ${(clientStats.displayed / elapsed).toFixed(1)} fps\n` +
                    `Decode queue: ${clientStats.queueSize}`;
            }
            clientStats.startedAt = now;
            clientStats.received = 0;
            clientStats.decoded = 0;
            clientStats.displayed = 0;
        };

        const updateServerStats = (message) => {
            const element = document.getElementById(`vtkweb-server-render-stats-${viewId}`);
            if (element) {
                element.textContent =
                    `Composite: ${(message.composite_fps || 0).toFixed(1)} fps\n` +
                    `Encode: ${(message.encoded_fps || 0).toFixed(1)} fps\n` +
                    `Delivered: ${(message.delivered_fps || 0).toFixed(1)} fps\n` +
                    `Data: ${(message.data_mib_s || 0).toFixed(2)} MiB/s`;
            }
            const ranks = document.getElementById(`vtkweb-server-render-ranks-${viewId}`);
            if (ranks) {
                ranks.textContent = (message.rank_fps || [])
                    .map((rank) => `Rank ${rank.rank}: ${(rank.fps || 0).toFixed(1)} fps`)
                    .join('\n');
            }
        };

        const renderActivities = (activities) => {
            const container = document.getElementById(`vtkweb-render-activities-${viewId}`);
            if (!container) return;
            const incoming = new Map((activities || []).map((activity) => [activity.key, activity]));

            for (const child of Array.from(container.children)) {
                if (!incoming.has(child.dataset.activityKey)) child.remove();
            }

            for (const activity of activities || []) {
                let card = Array.from(container.children).find(
                    (child) => child.dataset.activityKey === activity.key
                );
                if (!card) {
                    card = document.createElement('div');
                    card.dataset.activityKey = activity.key;
                    container.appendChild(card);
                }
                card.className = 'vtkweb-render-activity' +
                    (activity.complete ? ' complete' : '') +
                    (!activity.determinate && !activity.complete ? ' indeterminate' : '');
                card.innerHTML = '';

                const title = document.createElement('div');
                title.className = 'vtkweb-render-activity-title';
                const label = document.createElement('span');
                label.textContent = activity.label || '';
                const elapsed = document.createElement('span');
                elapsed.textContent = activity.complete ? '' : `${(activity.elapsed || 0).toFixed(2)}s`;
                title.append(label, elapsed);
                card.appendChild(title);

                const details = document.createElement('div');
                details.className = 'vtkweb-render-activity-details';
                for (const line of activity.details || []) {
                    const detail = document.createElement('div');
                    detail.textContent = line;
                    details.appendChild(detail);
                }
                card.appendChild(details);

                const progress = document.createElement('div');
                progress.className = 'vtkweb-render-activity-progress';
                const bar = document.createElement('div');
                bar.style.width = `${100 * (activity.complete ? 1 : (activity.progress || 0))}%`;
                progress.appendChild(bar);
                card.appendChild(progress);

                const values = document.createElement('div');
                values.className = 'vtkweb-render-activity-values';
                const list = activity.complete ? activity.duration_values : activity.progress_values;
                values.textContent = `[${(list || []).join('|')}]`;
                card.appendChild(values);
            }
        };

        const decoder = new VideoDecoder({
            output: (frame) => {
                const sequence = stream.pendingSequence;
                try {
                    clientStats.decoded += 1;
                    // VideoFrame remains browser/native memory. No copyTo(),
                    // getImageData(), or JS-visible pixel readback is performed.
                    ctx.drawImage(frame, 0, 0);
                    clientStats.displayed += 1;
                } finally {
                    frame.close();
                }

                clientStats.queueSize = decoder.decodeQueueSize;
                publishClientStats();

                // ACK only after decode + draw. The server keeps the encoded
                // access unit in flight until this arrives, and coalesces newer
                // raw framebuffer updates in the meantime.
                if (sequence !== null && socket.readyState === WebSocket.OPEN) {
                    socket.send(JSON.stringify({ type: 'ack', sequence }));
                    stream.pendingSequence = null;
                }
            },
            error: (error) => {
                console.error(`vtkweb H.264 decoder ${viewId}:`, error);
                socket.close();
            },
        });

        const stream = { socket, decoder, pendingSequence: null };
        streams.set(viewId, stream);

        socket.onopen = () => console.log(`vtkweb H.264 ${viewId}: video socket connected`);
        socket.onclose = (event) => {
            console.log(`vtkweb H.264 ${viewId}: video socket closed`, event.code, event.reason || '');
            if (streams.get(viewId) === stream) streams.delete(viewId);
            if (decoder.state !== 'closed') decoder.close();
        };
        socket.onerror = (error) => console.error(`vtkweb H.264 WebSocket ${viewId}:`, error);
        socket.onmessage = (event) => {
            if (typeof event.data === 'string') {
                const message = JSON.parse(event.data);
                if (message.type === 'error') {
                    console.error(`vtkweb video ${viewId}: ${message.message}`);
                    return;
                }
                if (message.type === 'stats') {
                    updateServerStats(message);
                    return;
                }
                if (message.type === 'activities') {
                    renderActivities(message.activities);
                    return;
                }
                if (message.type !== 'config') return;

                canvas.width = message.width;
                canvas.height = message.height;
                try {
                    if (decoder.state === 'configured') decoder.reset();
                    decoder.configure({
                        codec: message.codec,
                        codedWidth: message.codedWidth,
                        codedHeight: message.codedHeight,
                        optimizeForLatency: true,
                        hardwareAcceleration: 'prefer-hardware',
                    });
                } catch (error) {
                    console.error(`vtkweb cannot configure H.264 decoder ${viewId}:`, error);
                }
                return;
            }

            if (decoder.state !== 'configured') return;
            const bytes = new Uint8Array(event.data);
            if (bytes.byteLength <= 17) return;
            const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
            const keyframe = view.getUint8(0) !== 0;
            const sequence = Number(view.getBigUint64(1, false));
            const timestamp = Number(view.getBigUint64(9, false));
            const payload = bytes.subarray(17);

            clientStats.received += 1;
            stream.pendingSequence = sequence;
            decoder.decode(new EncodedVideoChunk({
                type: keyframe ? 'key' : 'delta',
                timestamp,
                data: payload,
            }));
            clientStats.queueSize = decoder.decodeQueueSize;
            publishClientStats();
        };
    };

    const reportRemoteSize = (element) => {
        const canvas = element.querySelector('canvas[id^="vtkweb-remote-frame-"]');
        const viewId = viewIdFromCanvas(canvas);
        if (!viewId) return;
        const rect = element.getBoundingClientRect();
        const width = Math.max(1, Math.round(rect.width));
        const height = Math.max(1, Math.round(rect.height));
        const previous = resizeTimers.get(viewId);
        if (previous !== undefined) window.clearTimeout(previous);
        resizeTimers.set(viewId, window.setTimeout(() => {
            resizeTimers.delete(viewId);
            window.__vtkwebSendRemoteResize?.(viewId, width, height);
        }, 150));
    };

    window.__vtkwebRequestRemoteSizes = () => {
        document.querySelectorAll('.vtkweb-remote-view').forEach(reportRemoteSize);
    };

    const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) reportRemoteSize(entry.target);
    });

    const syncRemoteViews = () => {
        const liveViews = new Set();
        document.querySelectorAll('.vtkweb-remote-view').forEach((element) => {
            const canvas = element.querySelector('canvas[id^="vtkweb-remote-frame-"]');
            const viewId = viewIdFromCanvas(canvas);
            if (!viewId) return;
            liveViews.add(viewId);
            connectCanvas(canvas);
            if (!observed.has(element)) {
                observed.add(element);
                resizeObserver.observe(element);
                reportRemoteSize(element);
            }
        });
        for (const viewId of streams.keys()) {
            if (!liveViews.has(viewId)) closeStream(viewId);
        }
    };

    const domObserver = new MutationObserver(syncRemoteViews);
    if (document.body) domObserver.observe(document.body, { childList: true, subtree: true });
    window.requestAnimationFrame(syncRemoteViews);

    window.__vtkwebShutdownRemoteVideo = () => {
        domObserver.disconnect();
        resizeObserver.disconnect();
        for (const timer of resizeTimers.values()) window.clearTimeout(timer);
        resizeTimers.clear();
        for (const viewId of Array.from(streams.keys())) closeStream(viewId);
        delete window.__vtkwebRequestRemoteSizes;
        delete window.__vtkwebShutdownRemoteVideo;
        delete window.__vtkwebH264Initialized;
    };
})();
        """
    )

    client.ClientTriggers(
        mounted="""
            window.__vtkwebTrigger = (name, args) => trigger(name, args);
            window.__vtkwebSendRemoteResize = (viewId, width, height) =>
                trigger('set_render_size', [viewId, width, height]);
            window.__vtkwebRequestRemoteSizes?.();

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
                    window.__vtkwebTrigger('set_split_ratio', [splitter.id, ratio]);
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
                    window.__vtkwebTrigger('interact_view_camera', [viewId, mode, dx, dy, height]);
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
                    // Keep wheel/trackpad zoom smoother than direct drag deltas.
                    window.__vtkwebTrigger('interact_view_camera', [viewId, 'zoom', 0, delta * 0.15, height]);
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
            window.__vtkwebShutdownRemoteVideo?.();
            delete window.__vtkwebTrigger;
            delete window.__vtkwebSendRemoteResize;
            delete window.__vtkwebStartTileResize;
            delete window.__vtkwebStartCameraDrag;
            delete window.__vtkwebRemoteWheel;
            delete window.__vtkwebRemoteWheelSessions;
        """,
    )

    with html.Div(classes="vtkweb-workspace"):
        # Render views are backend-agnostic H.264 presentation surfaces; dummy/empty views remain HTML.
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
                    classes="vtkweb-remote-frame",
                    id=("'vtkweb-remote-frame-' + tile.view_id",),
                )
                with html.Div(classes="vtkweb-render-stats"):
                    html.Div(
                        "",
                        id=("'vtkweb-server-render-stats-' + tile.view_id",),
                        classes="vtkweb-server-render-stats",
                    )
                    html.Div(
                        "",
                        id=("'vtkweb-client-video-stats-' + tile.view_id",),
                        classes="vtkweb-client-video-stats",
                    )
                    html.Div(
                        "",
                        id=("'vtkweb-server-render-ranks-' + tile.view_id",),
                        classes="vtkweb-render-stats-ranks",
                    )

                html.Div(
                    "",
                    id=("'vtkweb-render-activities-' + tile.view_id",),
                    classes="vtkweb-render-activities",
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
