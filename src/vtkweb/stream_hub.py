from __future__ import annotations

import asyncio
import json
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np
from aiohttp import web

from vtkweb.distributed import context


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
    clients: dict[int, web.WebSocketResponse] = field(default_factory=dict)
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
    stats_encoded_frames: int = 0
    stats_delivered_frames: int = 0
    stats_bytes: int = 0


@dataclass
class _RankActivity:
    generation: int = 0
    progress: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    duration: float | None = None
    done: bool = False


@dataclass
class _Activity:
    key: str
    view_id: str
    label: str
    determinate: bool
    ranks: dict[int, _RankActivity] = field(default_factory=dict)
    first_started_at: float = 0.0
    completed_at: float | None = None


class StreamHub:
    """CPU tile compositor + shared per-view PyAV/libx264 WebSocket stream.

    Every tile immediately updates the retained CPU framebuffer. At most one
    encoded H.264 access unit is in flight per view. Tile updates that arrive
    while it is being delivered are coalesced *before* x264; after delivery,
    only the newest framebuffer state is encoded. This preserves the H.264
    reference chain without delaying hot framebuffer updates.
    """

    route = "/vtkweb/video/{view_id}"
    _packet_header = struct.Struct("!BQQ")  # keyframe flag, sequence, timestamp_us

    def __init__(self, server) -> None:
        self._streams: dict[str, _ViewStream] = {}
        self._next_client_id = 1
        self._receiver_task: asyncio.Task | None = None
        self._activity_lock = threading.Lock()
        self._activity_events: deque[dict[str, Any]] = deque()
        self._activities: dict[str, _Activity] = {}
        server.controller.on_server_bind.add(self._on_server_bind)

    def _on_server_bind(self, http_server) -> None:
        http_server.app.router.add_get(self.route, self._handle_websocket)

    def start(self) -> None:
        """Start the one root-side receive loop when an event loop exists."""
        if not context.is_root:
            return
        if self._receiver_task is not None and not self._receiver_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._receiver_task = loop.create_task(self._receive_loop())

    def enqueue_activity(self, event: dict[str, Any]) -> None:
        """Thread-safe ingress for activities produced on rank 0 render threads."""
        with self._activity_lock:
            self._activity_events.append(dict(event))

    async def _receive_loop(self) -> None:
        """Drain local activities plus MPI frames/activities in one task."""
        while True:
            try:
                changed = self._drain_activities()

                if context.enabled:
                    for _ in range(32):
                        packet = context.poll_frame()
                        if packet is None:
                            break
                        await self.publish_frame(
                            packet["view_id"],
                            packet["rgb"],
                            width=int(packet["width"]),
                            height=int(packet["height"]),
                            region=tuple(packet["region"]),
                            full_size=tuple(packet["full_size"]),
                            size_revision=int(packet.get("size_revision", 0)),
                            source_rank=int(packet.get("source_rank", 0)),
                        )

                    while True:
                        event = context.poll_activity()
                        if event is None:
                            break
                        self._apply_activity(event)
                        changed = True

                if changed:
                    self._publish_activities()
                await asyncio.sleep(0.001 if context.enabled else 0.01)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[stream hub] receive failed: {exc}", flush=True)
                await asyncio.sleep(0.01)

    def _drain_activities(self) -> bool:
        changed = False
        while True:
            with self._activity_lock:
                event = self._activity_events.popleft() if self._activity_events else None
            if event is None:
                break
            self._apply_activity(event)
            changed = True
        return changed

    def _apply_activity(self, event: dict[str, Any]) -> None:
        key = str(event["key"])
        generation = int(event.get("generation", 0))
        rank = int(event.get("rank", 0))
        kind = str(event.get("kind", ""))
        activity = self._activities.get(key)

        if kind == "start":
            if activity is None:
                activity = _Activity(
                    key=key,
                    view_id=str(event.get("view_id", "")),
                    label=str(event.get("label", key)),
                    determinate=bool(event.get("determinate", True)),
                )
                self._activities[key] = activity
            current = activity.ranks.get(rank)
            if current is not None and generation < current.generation:
                return
            now = time.monotonic()
            if activity.first_started_at <= 0.0 or activity.completed_at is not None:
                activity.first_started_at = now
            activity.ranks[rank] = _RankActivity(
                generation=generation,
                progress=0.0 if activity.determinate else None,
                details=dict(event.get("details") or {}),
                started_at=now,
            )
            activity.completed_at = None
            return

        if activity is None:
            return
        rank_state = activity.ranks.get(rank)
        if rank_state is None or generation != rank_state.generation:
            return
        if kind == "progress":
            rank_state.progress = float(event.get("progress", 0.0))
        elif kind == "done":
            rank_state.progress = 1.0
            rank_state.duration = float(event.get("duration", 0.0))
            rank_state.done = True
            rank_state.details.update(event.get("details") or {})
            if all(
                activity.ranks.get(index) is not None
                and activity.ranks[index].done
                for index in range(context.size)
            ):
                activity.completed_at = time.monotonic()

    @staticmethod
    def _compact_activity_value(value: Any) -> str:
        if isinstance(value, (tuple, list)):
            values = [
                int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v
                for v in value
            ]
            if len(values) == 3 and values[0] == values[1] == values[2]:
                return f"{values[0]}³"
            return "×".join(str(v) for v in values)
        return str(value)

    def _activity_details(self, activity: _Activity) -> list[str]:
        names: list[str] = []
        for rank_state in activity.ranks.values():
            for name in rank_state.details:
                if name not in names:
                    names.append(name)
        lines = []
        for name in names:
            values = []
            present = []
            for rank in range(context.size):
                rank_state = activity.ranks.get(rank)
                if rank_state is None or name not in rank_state.details:
                    values.append("—")
                else:
                    value = self._compact_activity_value(rank_state.details[name])
                    values.append(value)
                    present.append(value)
            if present and len(present) == context.size and len(set(present)) == 1:
                lines.append(f"{name}={present[0]}")
            else:
                lines.append(f"{name}=[{'|'.join(values)}]")
        return lines

    def _publish_activities(self, view_id: str | None = None) -> None:
        now = time.monotonic()
        expired = [
            key
            for key, activity in self._activities.items()
            if activity.completed_at is not None and now - activity.completed_at > 3.0
        ]
        for key in expired:
            self._activities.pop(key, None)

        by_view: dict[str, list[dict[str, Any]]] = {}
        for activity in self._activities.values():
            if view_id is not None and activity.view_id != str(view_id):
                continue
            rank_states = [activity.ranks.get(rank) for rank in range(context.size)]
            complete = all(state is not None and state.done for state in rank_states)
            progresses = [
                0.0 if state is None or state.progress is None else float(state.progress)
                for state in rank_states
            ]
            card = {
                "key": activity.key,
                "view_id": activity.view_id,
                "label": activity.label,
                "determinate": activity.determinate,
                "progress": 1.0 if complete else sum(progresses) / float(max(1, context.size)),
                "progress_values": [
                    "—" if state is None or state.progress is None else f"{100.0 * state.progress:.0f}%"
                    for state in rank_states
                ],
                "duration_values": [
                    "—" if state is None or state.duration is None else f"{state.duration:.2f}s"
                    for state in rank_states
                ],
                "elapsed": max(0.0, now - activity.first_started_at),
                "complete": complete,
                "details": self._activity_details(activity),
            }
            by_view.setdefault(activity.view_id, []).append(card)

        if view_id is not None:
            target = str(view_id)
            self._publish_message(target, {"type": "activities", "activities": by_view.get(target, [])})
            return
        for target, cards in by_view.items():
            self._publish_message(target, {"type": "activities", "activities": cards})

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
        stream.stats_encoded_frames += 1
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
        self._publish_message(
            str(view_id),
            {
                "type": "stats",
                "rank_fps": rank_fps,
                "composite_fps": stream.stats_composite_frames / elapsed,
                "encoded_fps": stream.stats_encoded_frames / elapsed,
                "delivered_fps": stream.stats_delivered_frames / elapsed,
                "data_mib_s": stream.stats_bytes / elapsed / (1024.0 * 1024.0),
            },
        )

        stream.stats_started_at = now
        stream.stats_rank_frames.clear()
        stream.stats_composite_frames = 0
        stream.stats_encoded_frames = 0
        stream.stats_delivered_frames = 0
        stream.stats_bytes = 0


    def _publish_message(self, view_id: str, message: dict) -> None:
        """Send render telemetry directly over the per-view video socket.

        This intentionally bypasses Trame state so high-frequency telemetry
        cannot trigger Vue updates while a user is editing an unrelated widget.
        """
        stream = self._streams.get(str(view_id))
        if stream is None or not stream.clients:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        payload = json.dumps(message, separators=(",", ":"))
        loop.create_task(self._broadcast_text(stream, payload))

    @staticmethod
    async def _broadcast_text(stream: _ViewStream, payload: str) -> None:
        for ws in list(stream.clients.values()):
            if ws.closed:
                continue
            try:
                await ws.send_str(payload)
            except (ConnectionResetError, RuntimeError):
                pass

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
        stream.stats_encoded_frames = 0
        stream.stats_delivered_frames = 0
        stream.stats_bytes = 0

    @staticmethod
    async def _notify(stream: _ViewStream) -> None:
        async with stream.condition:
            stream.condition.notify_all()

    async def publish_frame(
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

        stream.stats_delivered_frames += 1
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
        stream.clients[client_id] = ws

        self.start()
        self._publish_activities(view_id=view_id)

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
                    frame.sequence,
                    frame.timestamp_us,
                )
                await ws.send_bytes(header + frame.payload)

                # Do not release this H.264 access unit until the browser has
                # decoded and drawn it. This bounds the browser decoder queue
                # without dropping inter-frame packets after x264.
                while not ws.closed:
                    message = await ws.receive()
                    if message.type == web.WSMsgType.TEXT:
                        try:
                            ack = json.loads(message.data)
                        except json.JSONDecodeError:
                            continue
                        if (
                            ack.get("type") == "ack"
                            and int(ack.get("sequence", -1)) == frame.sequence
                        ):
                            await self._mark_delivered(
                                stream, client_id, frame.sequence
                            )
                            break
                    elif message.type in {
                        web.WSMsgType.CLOSE,
                        web.WSMsgType.CLOSED,
                        web.WSMsgType.ERROR,
                    }:
                        break
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        except Exception as exc:
            if not ws.closed:
                await ws.send_str(json.dumps({"type": "error", "message": str(exc)}))
        finally:
            stream.clients.pop(client_id, None)
            # Do not let a disconnected client hold the in-flight access unit.
            if stream.encoded is not None:
                await self._mark_delivered(stream, client_id, stream.encoded.sequence)
            if not ws.closed:
                await ws.close()

        return ws


class StreamSink:
    """Producer-facing stream ingress used identically on every MPI rank."""

    def __init__(self, hub: StreamHub | None = None) -> None:
        self.hub = hub if context.is_root else None
        self._generations: dict[str, int] = {}

    def discard_view(self, view_id: str) -> None:
        if self.hub is not None:
            self.hub.discard_view(view_id)

    def reset_stats(self, view_id: str) -> None:
        if self.hub is not None:
            self.hub.reset_stats(view_id)

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
        source_rank: int | None = None,
    ) -> None:
        if not rgb:
            return
        rank = context.rank if source_rank is None else int(source_rank)
        kwargs = dict(
            width=int(width),
            height=int(height),
            region=region,
            full_size=full_size,
            size_revision=int(size_revision),
            source_rank=rank,
        )
        if context.is_root:
            if self.hub is not None:
                self.hub.start()
                await self.hub.publish_frame(view_id, rgb, **kwargs)
            return
        context.send_frame({"view_id": view_id, "rgb": rgb, **kwargs})

    def start(
        self,
        key: str,
        label: str,
        *,
        view_id: str,
        details: dict[str, Any] | None = None,
        determinate: bool = True,
    ) -> float:
        started_at = time.monotonic()
        key = str(key)
        generation = self._generations.get(key, 0) + 1
        self._generations[key] = generation
        self._activity_event({
            "kind": "start",
            "key": key,
            "generation": generation,
            "view_id": str(view_id),
            "label": str(label),
            "determinate": bool(determinate),
            "details": dict(details or {}),
            "started_at": started_at,
            "rank": context.rank,
        })
        return started_at

    def progress(self, key: str, progress: float) -> None:
        key = str(key)
        self._activity_event({
            "kind": "progress",
            "key": key,
            "generation": self._generations.get(key, 0),
            "progress": max(0.0, min(1.0, float(progress))),
            "rank": context.rank,
        })

    def done(
        self,
        key: str,
        started_at: float,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        key = str(key)
        self._activity_event({
            "kind": "done",
            "key": key,
            "generation": self._generations.get(key, 0),
            "duration": max(0.0, time.monotonic() - float(started_at)),
            "details": dict(details or {}),
            "rank": context.rank,
        })

    def _activity_event(self, event: dict[str, Any]) -> None:
        if context.is_root:
            if self.hub is not None:
                self.hub.enqueue_activity(event)
        else:
            context.send_activity(event)
