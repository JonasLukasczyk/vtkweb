from __future__ import annotations

import asyncio
import json
import struct
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Protocol

import numpy as np
from aiohttp import web


@dataclass
class _CompositeFrame:
    width: int
    height: int
    size_revision: int
    rgb: np.ndarray


@dataclass(frozen=True)
class _EncodedFrame:
    sequence: int
    logical_width: int
    logical_height: int
    coded_width: int
    coded_height: int
    keyframe: bool
    timestamp_us: int
    payload: bytes


@dataclass
class _ViewStream:
    """All mutable transport state for one logical view."""

    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    clients: set[int] = field(default_factory=set)
    pending_clients: set[int] = field(default_factory=set)
    frame: _CompositeFrame | None = None
    encoded: _EncodedFrame | None = None
    encoder: object | None = None
    encoder_size: tuple[int, int] | None = None
    sequence: int = 0
    frame_revision: int = 0
    encoded_revision: int = -1
    stats_started_at: float = field(default_factory=time.monotonic)
    stats_rank_frames: dict[int, int] = field(default_factory=dict)
    stats_known_ranks: set[int] = field(default_factory=set)
    stats_composite_frames: int = 0
    stats_bytes: int = 0


class FrameTransport(Protocol):
    """Transport interface for server-rendered RGB tiles."""

    def discard_view(self, view_id: str) -> None: ...

    async def publish(
        self,
        view_id: str,
        rgb: bytes,
        *,
        width: int,
        height: int,
        region=None,
        full_size=None,
        size_revision: int = 0,
        source_rank: int = 0,
    ) -> None: ...


class H264WebSocketFrameTransport:
    """CPU tile compositor + shared per-view PyAV/libx264 WebSocket stream.

    Every tile immediately updates the retained CPU framebuffer. At most one
    encoded H.264 access unit is in flight per view. Tile updates that arrive
    while it is being delivered are coalesced *before* x264; after delivery,
    only the newest framebuffer state is encoded. This preserves the H.264
    reference chain without delaying hot framebuffer updates.
    """

    route = "/vtkweb/video/{view_id}"
    _packet_header = struct.Struct("!BQ")  # keyframe flag, timestamp_us

    def __init__(self, server) -> None:
        self.server = server
        self._streams: dict[str, _ViewStream] = {}
        self.server.state.render_stats = {}
        self._next_client_id = 1
        server.controller.on_server_bind.add(self._on_server_bind)

    def _on_server_bind(self, http_server) -> None:
        http_server.app.router.add_get(self.route, self._handle_websocket)

    def _stream(self, view_id: str) -> _ViewStream:
        stream = self._streams.get(view_id)
        if stream is None:
            stream = _ViewStream()
            self._streams[view_id] = stream
        return stream

    @staticmethod
    def _make_encoder(width: int, height: int):
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("H.264 transport requires PyAV (pip install av).") from exc

        try:
            codec = av.CodecContext.create("libx264", "w")
        except Exception as exc:
            raise RuntimeError(
                "PyAV/FFmpeg does not provide the CPU libx264 encoder"
            ) from exc

        codec.width = width
        codec.height = height
        codec.pix_fmt = "yuv420p"
        codec.time_base = Fraction(1, 1_000_000)
        codec.framerate = Fraction(60, 1)
        codec.gop_size = 60
        codec.max_b_frames = 0
        codec.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "crf": "20",
            "x264-params": "repeat-headers=1:annexb=1:scenecut=0",
        }
        codec.open()
        return codec

    def _encoder(self, stream: _ViewStream, width: int, height: int):
        size = (width, height)
        if stream.encoder is None or stream.encoder_size != size:
            stream.encoder = self._make_encoder(width, height)
            stream.encoder_size = size
        return stream.encoder

    def _encode(
        self,
        stream: _ViewStream,
        frame: _CompositeFrame,
        sequence: int,
    ) -> _EncodedFrame | None:
        import av

        logical_width = frame.width
        logical_height = frame.height
        coded_width = logical_width + (logical_width & 1)
        coded_height = logical_height + (logical_height & 1)

        rgb = frame.rgb
        if coded_width != logical_width or coded_height != logical_height:
            padded = np.zeros((coded_height, coded_width, 3), dtype=np.uint8)
            padded[:logical_height, :logical_width] = rgb
            rgb = padded

        encoder = self._encoder(stream, coded_width, coded_height)
        video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        timestamp_us = time.monotonic_ns() // 1000
        video_frame.pts = timestamp_us
        video_frame.time_base = Fraction(1, 1_000_000)
        packets = encoder.encode(video_frame)
        if not packets:
            return None

        return _EncodedFrame(
            sequence=sequence,
            logical_width=logical_width,
            logical_height=logical_height,
            coded_width=coded_width,
            coded_height=coded_height,
            keyframe=any(packet.is_keyframe for packet in packets),
            timestamp_us=timestamp_us,
            payload=b"".join(bytes(packet) for packet in packets),
        )

    def _encode_latest(self, stream: _ViewStream) -> bool:
        if stream.encoded is not None or not stream.clients or stream.frame is None:
            return False
        if stream.frame_revision <= stream.encoded_revision:
            return False

        sequence = stream.sequence + 1
        encoded = self._encode(stream, stream.frame, sequence)
        if encoded is None:
            return False

        stream.sequence = sequence
        stream.encoded_revision = stream.frame_revision
        stream.encoded = encoded
        # Only clients present at encode time must consume this access unit.
        stream.pending_clients = set(stream.clients)
        return True


    def _record_stats(self, view_id: str, stream: _ViewStream, source_rank: int, byte_count: int) -> None:
        rank = int(source_rank)
        stream.stats_known_ranks.add(rank)
        stream.stats_rank_frames[rank] = stream.stats_rank_frames.get(rank, 0) + 1
        stream.stats_composite_frames += 1
        stream.stats_bytes += int(byte_count)

        now = time.monotonic()
        elapsed = now - stream.stats_started_at
        if elapsed < 1.0:
            return

        rank_fps = [
            {"rank": rank_id, "fps": stream.stats_rank_frames.get(rank_id, 0) / elapsed}
            for rank_id in sorted(stream.stats_known_ranks)
        ]
        current = dict(self.server.state.render_stats or {})
        current[str(view_id)] = {
            "rank_fps": rank_fps,
            "composite_fps": stream.stats_composite_frames / elapsed,
            "data_mib_s": stream.stats_bytes / elapsed / (1024.0 * 1024.0),
        }
        self.server.state.render_stats = current
        # Stats are produced outside a Trame RPC callback. Publish only this key
        # instead of calling state.flush(), which would also push unrelated
        # pending UI state and can overwrite controls while the user is editing.
        protocol = self.server.protocol
        if protocol is not None:
            payload = self.server.state.translator.translate_dict(
                {"render_stats": current}
            )
            protocol.push_state_change(payload)
            # Mark this one key as committed so a later normal Trame flush does
            # not resend it together with unrelated state.
            self.server.state.clean("render_stats")

        stream.stats_started_at = now
        stream.stats_rank_frames.clear()
        stream.stats_composite_frames = 0
        stream.stats_bytes = 0

    def reset_stats(self, view_id: str) -> None:
        """Start a fresh measurement window for one view.

        Used when switching render modes so the next sample measures only the
        new mode instead of averaging across both kernels.
        """
        stream = self._streams.get(str(view_id))
        if stream is None:
            return
        stream.stats_started_at = time.monotonic()
        stream.stats_rank_frames.clear()
        stream.stats_composite_frames = 0
        stream.stats_bytes = 0

    @staticmethod
    async def _notify(stream: _ViewStream) -> None:
        async with stream.condition:
            stream.condition.notify_all()

    async def publish(
        self,
        view_id: str,
        rgb: bytes,
        *,
        width: int,
        height: int,
        region=None,
        full_size=None,
        size_revision: int = 0,
        source_rank: int = 0,
    ) -> None:
        if not rgb:
            return

        tile_width = max(1, int(width))
        tile_height = max(1, int(height))
        expected = tile_width * tile_height * 3
        if len(rgb) != expected:
            raise ValueError(f"RGB payload has {len(rgb)} bytes, expected {expected}")

        if full_size is None:
            full_width, full_height = tile_width, tile_height
        else:
            full_width, full_height = map(int, full_size)

        if region is None:
            x, y = 0, 0
        else:
            x, y, region_width, region_height = map(int, region)
            if (region_width, region_height) != (tile_width, tile_height):
                raise ValueError("Tile dimensions do not match its region")

        if x < 0 or y < 0 or x + tile_width > full_width or y + tile_height > full_height:
            raise ValueError("Tile region lies outside the full framebuffer")

        stream = self._stream(str(view_id))
        revision = max(1, int(size_revision))
        frame = stream.frame
        if frame is not None and revision < frame.size_revision:
            return
        if (
            frame is None
            or frame.size_revision != revision
            or frame.width != full_width
            or frame.height != full_height
        ):
            frame = _CompositeFrame(
                width=full_width,
                height=full_height,
                size_revision=revision,
                rgb=np.zeros((full_height, full_width, 3), dtype=np.uint8),
            )
            stream.frame = frame

        tile_rgb = np.frombuffer(rgb, dtype=np.uint8).reshape(tile_height, tile_width, 3)
        frame.rgb[y : y + tile_height, x : x + tile_width] = tile_rgb
        stream.frame_revision += 1
        self._record_stats(view_id, stream, source_rank, len(rgb))

        # If an H.264 access unit is already in flight, just keep accumulating
        # the newest raw state. Encoding it now would force us to queue or drop
        # an inter-frame packet, both of which are undesirable here.
        if stream.encoded is not None or not stream.clients:
            return

        if self._encode_latest(stream):
            await self._notify(stream)

    async def _mark_delivered(
        self,
        stream: _ViewStream,
        client_id: int,
        sequence: int,
    ) -> None:
        current = stream.encoded
        if current is None or current.sequence != sequence:
            return

        stream.pending_clients.discard(client_id)
        if stream.pending_clients:
            return

        stream.encoded = None
        if self._encode_latest(stream):
            await self._notify(stream)

    async def _wait_for_encoded(
        self,
        stream: _ViewStream,
        after_sequence: int,
    ) -> _EncodedFrame:
        async with stream.condition:
            while stream.encoded is None or stream.encoded.sequence <= after_sequence:
                await stream.condition.wait()
            return stream.encoded

    def discard_view(self, view_id: str) -> None:
        stream = self._streams.get(str(view_id))
        if stream is None:
            return

        # Resize/scene reset: drop image/codec state but preserve connected
        # clients, their condition, and the monotonic stream sequence.
        stream.frame = None
        stream.encoded = None
        stream.pending_clients.clear()
        stream.encoder = None
        stream.encoder_size = None
        stream.frame_revision = 0
        stream.encoded_revision = -1

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        view_id = str(request.match_info.get("view_id") or "")
        if not view_id:
            raise web.HTTPBadRequest(text="view_id is required")

        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)

        stream = self._stream(view_id)
        client_id = self._next_client_id
        self._next_client_id += 1
        stream.clients.add(client_id)

        # A new WebCodecs decoder must start on a keyframe. Restarting the
        # shared encoder makes the next access unit a fresh GOP for the newcomer.
        stream.encoder = None
        stream.encoder_size = None

        sequence = stream.sequence
        configured_size: tuple[int, int] | None = None
        needs_keyframe = True

        try:
            while not ws.closed:
                frame = await self._wait_for_encoded(stream, sequence)
                sequence = frame.sequence
                size = (frame.logical_width, frame.logical_height)

                if configured_size != size:
                    configured_size = size
                    needs_keyframe = True
                    await ws.send_str(
                        json.dumps(
                            {
                                "type": "config",
                                "codec": "avc1.42E033",
                                "codedWidth": frame.coded_width,
                                "codedHeight": frame.coded_height,
                                "width": frame.logical_width,
                                "height": frame.logical_height,
                            },
                            separators=(",", ":"),
                        )
                    )

                if needs_keyframe and not frame.keyframe:
                    await self._mark_delivered(stream, client_id, frame.sequence)
                    continue
                needs_keyframe = False

                header = self._packet_header.pack(
                    1 if frame.keyframe else 0,
                    frame.timestamp_us,
                )
                await ws.send_bytes(header + frame.payload)
                await self._mark_delivered(stream, client_id, frame.sequence)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        except Exception as exc:
            if not ws.closed:
                await ws.send_str(json.dumps({"type": "error", "message": str(exc)}))
        finally:
            stream.clients.discard(client_id)
            # Do not let a disconnected client hold the in-flight access unit.
            if stream.encoded is not None:
                await self._mark_delivered(stream, client_id, stream.encoded.sequence)
            if not ws.closed:
                await ws.close()

        return ws
