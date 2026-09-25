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
    window.__vtkwebRemoteGpuViews = new Map();
    window.__vtkwebRemoteGpuDevicePromise = null;

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

    const decodeDepth = async (bytes, encoding) => {
        if (!bytes || bytes.byteLength === 0) return null;
        if (encoding !== 'zlib-f32-linear-le') return null;
        if (typeof DecompressionStream === 'undefined') return null;
        const stream = new Blob([bytes]).stream().pipeThrough(
            new DecompressionStream('deflate')
        );
        const raw = await new Response(stream).arrayBuffer();
        if ((raw.byteLength % 4) !== 0) return null;
        // The server emits little-endian float32 values. All supported browser
        // targets are little-endian.
        return new Float32Array(raw);
    };

    const getRemoteGpuDevice = async () => {
        if (!navigator.gpu) return null;
        if (!window.__vtkwebRemoteGpuDevicePromise) {
            window.__vtkwebRemoteGpuDevicePromise = (async () => {
                const adapter = await navigator.gpu.requestAdapter();
                return adapter ? adapter.requestDevice() : null;
            })().catch((error) => {
                console.warn('vtkweb WebGPU initialization failed', error);
                return null;
            });
        }
        return window.__vtkwebRemoteGpuDevicePromise;
    };

    const alignTo = (value, alignment) => Math.ceil(value / alignment) * alignment;

    const makeDepthUpload = (depth, width, height) => {
        const rowBytes = width * 4;
        const bytesPerRow = alignTo(rowBytes, 256);
        if (bytesPerRow === rowBytes) {
            return { data: new Uint8Array(depth.buffer, depth.byteOffset, depth.byteLength), bytesPerRow };
        }
        const padded = new Uint8Array(bytesPerRow * height);
        const source = new Uint8Array(depth.buffer, depth.byteOffset, depth.byteLength);
        for (let row = 0; row < height; row += 1) {
            padded.set(
                source.subarray(row * rowBytes, (row + 1) * rowBytes),
                row * bytesPerRow,
            );
        }
        return { data: padded, bytesPerRow };
    };

    const initializeDepthTexture = (device, texture, width, height) => {
        const rowBytes = width * 4;
        const bytesPerRow = alignTo(rowBytes, 256);
        const row = new Float32Array(bytesPerRow / 4);
        row.fill(0.0);
        const data = new Uint8Array(bytesPerRow * height);
        const rowBytesView = new Uint8Array(row.buffer);
        for (let y = 0; y < height; y += 1) data.set(rowBytesView, y * bytesPerRow);
        device.queue.writeTexture(
            { texture }, data,
            { bytesPerRow, rowsPerImage: height },
            { width, height, depthOrArrayLayers: 1 },
        );
    };

    const ensureGpuView = async (canvas, viewId, width, height) => {
        const device = await getRemoteGpuDevice();
        if (!device) return null;
        let state = window.__vtkwebRemoteGpuViews.get(viewId);
        if (state && state.canvas === canvas && state.width === width && state.height === height) {
            return state;
        }
        if (state) {
            state.colorTexture?.destroy();
            state.depthTexture?.destroy();
        }
        const context = canvas.getContext('webgpu');
        if (!context) return null;
        const format = navigator.gpu.getPreferredCanvasFormat();
        context.configure({ device, format, alphaMode: 'opaque' });
        const colorTexture = device.createTexture({
            size: [width, height, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
        });
        const depthTexture = device.createTexture({
            size: [width, height, 1],
            format: 'r32float',
            usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
        });
        initializeDepthTexture(device, depthTexture, width, height);
        const uniformBuffer = device.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const shader = device.createShaderModule({ code: `
            struct Params {
                debug: u32,
                width: u32,
                height: u32,
                ssaoSlices: u32,
                ssaoSteps: u32,
                depthNear: f32,
                depthFar: f32,
                ssaoRadius: f32,
                ssaoStrength: f32,
                ssaoThickness: f32,
                cameraFovRadians: f32,
                pad0: f32,
            };
            @group(0) @binding(0) var colorTex: texture_2d<f32>;
            @group(0) @binding(1) var depthTex: texture_2d<f32>;
            @group(0) @binding(2) var<uniform> params: Params;

            const PI: f32 = 3.141592653589793;
            const SECTOR_COUNT: u32 = 32u;
            const FULL_MASK: u32 = 0xffffffffu;

            @vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
                var pos = array<vec2<f32>, 3>(
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>( 3.0, -1.0),
                    vec2<f32>(-1.0,  3.0)
                );
                return vec4<f32>(pos[index], 0.0, 1.0);
            }

            fn loadDepth(coord: vec2<i32>) -> f32 {
                let cx = clamp(coord.x, 0, i32(params.width) - 1);
                let cy = clamp(coord.y, 0, i32(params.height) - 1);
                let value = textureLoad(depthTex, vec2<i32>(cx, cy), 0).r;
                return select(params.depthFar, value, value > 0.0);
            }

            fn readDepthLinear(pixel: vec2<f32>) -> f32 {
                let base = vec2<i32>(floor(pixel));
                let frac = fract(pixel);
                let d00 = loadDepth(base);
                let d10 = loadDepth(base + vec2<i32>(1, 0));
                let d01 = loadDepth(base + vec2<i32>(0, 1));
                let d11 = loadDepth(base + vec2<i32>(1, 1));
                let d0 = mix(d00, d10, frac.x);
                let d1 = mix(d01, d11, frac.x);
                return mix(d0, d1, frac.y);
            }

            fn rayDirection(pixel: vec2<f32>) -> vec3<f32> {
                let size = vec2<f32>(f32(params.width), f32(params.height));
                let ndc = pixel / size * 2.0 - vec2<f32>(1.0);
                let tanHalfFov = tan(max(params.cameraFovRadians * 0.5, 1.0e-5));
                let aspect = size.x / max(size.y, 1.0);
                return normalize(vec3<f32>(ndc.x * aspect * tanHalfFov,
                                           -ndc.y * tanHalfFov, -1.0));
            }

            fn viewPosition(pixel: vec2<f32>, depth: f32) -> vec3<f32> {
                return rayDirection(pixel) * depth;
            }

            fn reconstructNormal(pixel: vec2<f32>, centerDepth: f32) -> vec3<f32> {
                let p = viewPosition(pixel, centerDepth);
                let dl = readDepthLinear(pixel - vec2<f32>(1.0, 0.0));
                let dr = readDepthLinear(pixel + vec2<f32>(1.0, 0.0));
                let du = readDepthLinear(pixel - vec2<f32>(0.0, 1.0));
                let dd = readDepthLinear(pixel + vec2<f32>(0.0, 1.0));
                let pl = viewPosition(pixel - vec2<f32>(1.0, 0.0), dl);
                let pr = viewPosition(pixel + vec2<f32>(1.0, 0.0), dr);
                let pu = viewPosition(pixel - vec2<f32>(0.0, 1.0), du);
                let pd = viewPosition(pixel + vec2<f32>(0.0, 1.0), dd);
                let dx = select(p - pl, pr - p, abs(dr - centerDepth) < abs(dl - centerDepth));
                let dy = select(p - pu, pd - p, abs(dd - centerDepth) < abs(du - centerDepth));
                var n = normalize(cross(dx, dy));
                let viewDir = normalize(-p);
                if (dot(n, viewDir) < 0.0) { n = -n; }
                return n;
            }

            fn hash12(p: vec2<f32>) -> f32 {
                let h = dot(p, vec2<f32>(127.1, 311.7));
                return fract(sin(h) * 43758.5453123);
            }

            fn angleBitMask(start: u32, width: u32) -> u32 {
                if (width == 0u) { return 0u; }
                if (width >= SECTOR_COUNT) { return FULL_MASK; }
                return (FULL_MASK >> (SECTOR_COUNT - width)) << start;
            }

            // Cosine-weighted slice-relative CDF remap from GT-VBAO. Equal-width
            // bits then represent equal cosine-weighted portions of the slice.
            fn remapHorizonCosine(horizonCos: f32, directionRight: bool,
                                  normalAngle: f32, cosN: f32, sinN: f32) -> f32 {
                let d = select(-1.0, 1.0, directionRight);
                let h = max(clamp(horizonCos, -1.0, 1.0), sinN * d);
                let sinTheta = sqrt(max(1.0 - h * h, 0.0));
                let theta = acos(clamp(h, -1.0, 1.0));
                var n0 = 3.0;
                var n1 = -1.0;
                var n2 = 4.0;
                if (directionRight) {
                    n0 = 1.0;
                    n1 = 1.0;
                    n2 = 0.0;
                }
                let cosTwoPsiPlusN = (2.0 * h * h - 1.0) * cosN
                    - h * sinTheta * (2.0 * d) * sinN;
                let t0 = cosN * n0 + cosTwoPsiPlusN * n1
                    + (theta * (2.0 * n1 * d)
                       + normalAngle * (n2 + 2.0 * n1) - PI) * sinN;
                let invNormalization = 0.25 / max(cosN + normalAngle * sinN, 1.0e-4);
                return clamp(t0 * invNormalization, 0.0, 1.0);
            }

            fn gtvbao(center: vec2<f32>, centerDepth: f32) -> f32 {
                let sliceCount = min(params.ssaoSlices, 8u);
                let stepCount = min(max(params.ssaoSteps, 1u), 32u);
                if (sliceCount == 0u || params.ssaoRadius <= 0.0 ||
                    params.ssaoStrength <= 0.0 || centerDepth >= params.depthFar) {
                    return 1.0;
                }

                let centerPos = viewPosition(center, centerDepth);
                let viewDir = normalize(-centerPos);
                let viewNormal = reconstructNormal(center, centerDepth);

                // Minimal-rotation frame around the view vector, then perspective-
                // project each slice direction into screen space (GT-VBAO).
                let frameA = 1.0 / max(viewDir.z + 1.0, 1.0e-5);
                let frameB = -viewDir.x * viewDir.y * frameA;
                let frameTangent = vec3<f32>(1.0 - viewDir.x * viewDir.x * frameA,
                                             frameB, -viewDir.x);
                let frameBitangent = vec3<f32>(frameB,
                                               1.0 - viewDir.y * viewDir.y * frameA,
                                               -viewDir.y);

                let focalPixels = 0.5 * f32(params.height) /
                    tan(max(params.cameraFovRadians * 0.5, 1.0e-5));
                let radiusPixels = params.ssaoRadius * focalPixels / max(centerDepth, 1.0e-5);
                if (radiusPixels <= 0.5) { return 1.0; }

                var weightedOcclusion = 0.0;
                var sliceWeightSum = 0.0;
                let pixelNoise = hash12(center);

                for (var s: u32 = 0u; s < 8u; s = s + 1u) {
                    if (s >= sliceCount) { break; }
                    let angle = (f32(s) + pixelNoise) * PI / f32(sliceCount);
                    let direction = vec2<f32>(cos(angle), sin(angle));
                    let frameDirection = frameTangent * direction.x + frameBitangent * direction.y;
                    let projected = frameDirection.xy * viewDir.z - viewDir.xy * frameDirection.z;
                    if (dot(projected, projected) < 1.0e-8) { continue; }
                    let screenDirection = normalize(projected);
                    let sliceDir = vec3<f32>(screenDirection, 0.0);
                    let planeNormal = normalize(cross(sliceDir, viewDir));
                    let tangent = cross(viewDir, planeNormal);
                    let projectedNormal = viewNormal - planeNormal * dot(viewNormal, planeNormal);
                    let projectedNormalLength = length(projectedNormal);
                    if (projectedNormalLength < 1.0e-4) { continue; }
                    let projectedNormalN = projectedNormal / projectedNormalLength;
                    let cosN = clamp(dot(projectedNormalN, viewDir), -1.0, 1.0);
                    let normalSign = -sign(dot(projectedNormal, tangent));
                    let normalAngle = normalSign * acos(cosN);
                    let sinN = normalSign * sqrt(max(1.0 - cosN * cosN, 0.0));
                    let quantizeDither = hash12(center + vec2<f32>(f32(s) * 19.19, 7.73));
                    var occluded: u32 = 0u;

                    for (var directionIndex: u32 = 0u; directionIndex < 2u; directionIndex = directionIndex + 1u) {
                        let directionRight = directionIndex == 0u;
                        let directionSign = select(-1.0, 1.0, directionRight);
                        for (var i: u32 = 0u; i < 32u; i = i + 1u) {
                            if (i >= stepCount) { break; }
                            let fi = f32(i);
                            let denom = max(f32(stepCount - 1u), 1.0);
                            let jitter = hash12(center + vec2<f32>(f32(s) * 31.0 + f32(directionIndex) * 13.0,
                                                                  f32(i) * 5.0));
                            let u = clamp((fi + 0.35 + 0.3 * jitter) / denom, 0.0, 1.0);
                            let distancePixels = max(u * u * radiusPixels, fi + 1.0);
                            let samplePixel = center + screenDirection * (directionSign * distancePixels);
                            if (samplePixel.x <= 0.0 || samplePixel.y <= 0.0 ||
                                samplePixel.x >= f32(params.width - 1u) ||
                                samplePixel.y >= f32(params.height - 1u)) {
                                break;
                            }
                            let sampleDepth = readDepthLinear(samplePixel);
                            let samplePos = viewPosition(samplePixel, sampleDepth);
                            let v = samplePos - centerPos;
                            let distanceSq = max(dot(v, v), 1.0e-8);
                            let invDistance = inverseSqrt(distanceSq);
                            let distance = distanceSq * invDistance;
                            let projectedDistance = dot(v, viewDir);
                            let sampleThickness = min(params.ssaoThickness, distance * 0.8);
                            let frontHorizon = projectedDistance * invDistance;
                            let backVector = v - viewDir * sampleThickness;
                            let backHorizon = (projectedDistance - sampleThickness) *
                                inverseSqrt(max(dot(backVector, backVector), 1.0e-8));
                            var front = remapHorizonCosine(frontHorizon, directionRight,
                                                           normalAngle, cosN, sinN);
                            var back = remapHorizonCosine(backHorizon, directionRight,
                                                          normalAngle, cosN, sinN);
                            if (directionRight) {
                                let tmp = front;
                                front = back;
                                back = tmp;
                            }
                            let minH = min(front, back);
                            let maxH = max(front, back);
                            let start = u32(clamp(floor(minH * 32.0 + quantizeDither), 0.0, 32.0));
                            let end = u32(clamp(floor(maxH * 32.0 + quantizeDither), 0.0, 32.0));
                            if (end > start) {
                                occluded = occluded | angleBitMask(start, end - start);
                            }
                            if (occluded == FULL_MASK) { break; }
                        }
                    }

                    let sliceOcclusion = f32(countOneBits(occluded)) / 32.0;
                    let sliceWeight = max(projectedNormalLength *
                        (cosN + normalAngle * sinN), 0.0);
                    weightedOcclusion = weightedOcclusion + sliceOcclusion * sliceWeight;
                    sliceWeightSum = sliceWeightSum + sliceWeight;
                }

                let occlusion = clamp(weightedOcclusion / max(sliceWeightSum, 1.0e-4), 0.0, 1.0);
                return clamp(pow(max(1.0 - occlusion, 0.0), params.ssaoStrength), 0.0, 1.0);
            }

            @fragment fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
                let x = clamp(i32(position.x), 0, i32(params.width) - 1);
                let y = clamp(i32(position.y), 0, i32(params.height) - 1);
                let coord = vec2<i32>(x, y);
                let depth = loadDepth(coord);
                if (params.debug != 0u) {
                    let depthWidth = max(params.depthFar - params.depthNear, 1.0e-12);
                    let normalized = clamp((depth - params.depthNear) / depthWidth, 0.0, 1.0);
                    let value = 1.0 - normalized;
                    return vec4<f32>(value, value, value, 1.0);
                }
                let rgba = textureLoad(colorTex, coord, 0);
                let ao = gtvbao(vec2<f32>(f32(x) + 0.5, f32(y) + 0.5), depth);
                return vec4<f32>(rgba.rgb * ao, rgba.a);
            }
        ` });
        const pipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: shader, entryPoint: 'vs' },
            fragment: { module: shader, entryPoint: 'fs', targets: [{ format }] },
            primitive: { topology: 'triangle-list' },
        });
        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: colorTexture.createView() },
                { binding: 1, resource: depthTexture.createView() },
                { binding: 2, resource: { buffer: uniformBuffer } },
            ],
        });
        state = {
            device, canvas, context, format, width, height,
            colorTexture, depthTexture, uniformBuffer, pipeline, bindGroup,
            debug: false, depthNear: 0.0, depthFar: 1.0,
            ssaoSlices: 0, ssaoSteps: 6, ssaoRadius: 10.0, ssaoStrength: 1.0,
            ssaoThickness: 0.4, cameraFov: 30.0,
        };
        window.__vtkwebRemoteGpuViews.set(viewId, state);
        return state;
    };

    const presentGpuView = (state) => {
        const params = new ArrayBuffer(48);
        const ints = new Uint32Array(params);
        const floats = new Float32Array(params);
        ints[0] = state.debug ? 1 : 0;
        ints[1] = state.width;
        ints[2] = state.height;
        ints[3] = Math.max(0, Math.min(8, Number(state.ssaoSlices || 0)));
        ints[4] = Math.max(1, Math.min(32, Number(state.ssaoSteps || 6)));
        floats[5] = state.depthNear;
        floats[6] = state.depthFar;
        floats[7] = state.ssaoRadius;
        floats[8] = state.ssaoStrength;
        floats[9] = state.ssaoThickness;
        floats[10] = state.cameraFov * Math.PI / 180.0;
        floats[11] = 0.0;
        state.device.queue.writeBuffer(state.uniformBuffer, 0, params);
        const encoder = state.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: state.context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });
        pass.setPipeline(state.pipeline);
        pass.setBindGroup(0, state.bindGroup);
        pass.draw(3);
        pass.end();
        state.device.queue.submit([encoder.finish()]);
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

        const payloadOffset = 4 + headerLength;
        const depthLength = Number(header.depth_length || 0);
        const imageLength = Number(
            header.image_length || Math.max(0, packetBytes.byteLength - payloadOffset - depthLength)
        );
        if (payloadOffset + imageLength + depthLength > packetBytes.byteLength) return null;
        const imageBytes = packetBytes.slice(payloadOffset, payloadOffset + imageLength);
        const depthBytes = depthLength > 0
            ? packetBytes.slice(payloadOffset + imageLength, payloadOffset + imageLength + depthLength)
            : null;
        const blob = new Blob([imageBytes], { type: header.mime_type || 'image/jpeg' });

        let bitmap;
        let depth = null;
        try {
            [bitmap, depth] = await Promise.all([
                createImageBitmap(blob),
                decodeDepth(depthBytes, header.depth_encoding),
            ]);
        } catch (_error) {
            return null;
        }

        window.__vtkwebRemoteLatestFrame.set(
            frameKey,
            { generation, sequence, sizeRevision },
        );
        return { header, bitmap, depth, viewId, tileId };
    };

    const drawRemoteBatch = async (tiles) => {
        const touchedViews = new Set();
        const targetRevision = new Map(window.__vtkwebRemoteSizeRevision);
        for (const tile of tiles) {
            if (!tile) continue;
            const revision = Number(tile.header.size_revision || 0);
            const previous = Number(targetRevision.get(tile.viewId) || 0);
            if (revision > previous) targetRevision.set(tile.viewId, revision);
        }

        for (const tile of tiles) {
            if (!tile) continue;
            const { header, bitmap, depth, viewId } = tile;
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

            const x = region ? Number(region[0]) : 0;
            const y = region ? Number(region[1]) : 0;
            const width = region ? Number(region[2]) : bitmap.width;
            const height = region ? Number(region[3]) : bitmap.height;
            const gpu = await ensureGpuView(canvas, viewId, targetWidth, targetHeight);
            if (gpu) {
                gpu.device.queue.copyExternalImageToTexture(
                    { source: bitmap },
                    { texture: gpu.colorTexture, origin: { x, y, z: 0 } },
                    { width, height, depthOrArrayLayers: 1 },
                );
                if (depth && depth.length === width * height) {
                    if (Boolean(header.debug)) {
                        let validCount = 0;
                        let minDepth = Infinity;
                        let maxDepth = -Infinity;
                        for (let i = 0; i < depth.length; i += 1) {
                            const value = depth[i];
                            if (Number.isFinite(value) && value > 0.0) {
                                validCount += 1;
                                minDepth = Math.min(minDepth, value);
                                maxDepth = Math.max(maxDepth, value);
                            }
                        }
                        console.debug(
                            '[vtkweb depth tile]',
                            {
                                viewId, tileId: tile.tileId, validCount, total: depth.length,
                                minDepth: validCount ? minDepth : null,
                                maxDepth: validCount ? maxDepth : null,
                                depthNear: Number(header.depth_near || 0.0),
                                depthFar: Number(header.depth_far || 1.0),
                            },
                        );
                    }
                    const upload = makeDepthUpload(depth, width, height);
                    gpu.device.queue.writeTexture(
                        { texture: gpu.depthTexture, origin: { x, y, z: 0 } },
                        upload.data,
                        { bytesPerRow: upload.bytesPerRow, rowsPerImage: height },
                        { width, height, depthOrArrayLayers: 1 },
                    );
                    gpu.depthNear = Number(header.depth_near || 0.0);
                    gpu.depthFar = Number(header.depth_far || 1.0);
                }
                gpu.debug = Boolean(header.debug);
                gpu.ssaoSlices = Number(header.ssao_slices || 0);
                gpu.ssaoSteps = Number(header.ssao_steps || 6);
                gpu.ssaoRadius = Number(header.ssao_radius || 10.0);
                gpu.ssaoStrength = Number(header.ssao_strength || 1.0);
                gpu.ssaoThickness = Number(header.ssao_thickness || 0.4);
                gpu.cameraFov = Number(header.camera_fov || 30.0);
                touchedViews.add(viewId);
            } else {
                const context = canvas.getContext('2d', { alpha: false });
                if (context) {
                    context.drawImage(bitmap, x, y);
                    touchedViews.add(viewId);
                }
            }
            bitmap.close();
        }

        for (const viewId of touchedViews) {
            const gpu = window.__vtkwebRemoteGpuViews.get(viewId);
            if (gpu) presentGpuView(gpu);
        }

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
                Promise.resolve(drawRemoteBatch(decoded)).finally(resolve);
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
