from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import dataclass, field
from typing import Protocol

from aiohttp import WSMsgType, web

_BATCH_MAGIC = b"VTB1"


@dataclass
class _CachedTile:
    packet: bytes
    revision: int


@dataclass
class _Client:
    websocket: web.WebSocketResponse
    credit: bool = False
    last_sent: dict[tuple[str, int], int] = field(default_factory=dict)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    flush_task: asyncio.Task | None = None


class FrameTransport(Protocol):
    """Transport interface for encoded server-rendered frames."""

    async def publish(
        self,
        view_id: str,
        image: bytes,
        *,
        mime_type: str = "image/jpeg",
        generation: int = 0,
        sequence: int = 0,
        region=None,
        full_size=None,
        tile_id: int = 0,
        debug: bool = False,
        size_revision: int = 0,
    ) -> None: ...


class WebSocketFrameTransport:
    """Client-paced latest-value binary frame stream.

    Rank 0 stores only the latest encoded packet for each ``(view_id, tile_id)``.
    A browser grants one delivery credit after it has painted the previous batch.
    One credit sends one batch containing only tiles that changed since that
    client's previous batch. A short coalescing window lets tiles from multiple
    ranks and views accumulate before the credit is consumed. If nothing has
    changed yet, the credit remains outstanding until a later publish schedules
    another coalesced flush.

    Batch wire format::

        b"VTB1" | uint32_be count |
            uint32_be packet_length | packet |
            ...

    Each embedded packet retains the original tile format::

        uint32_be header_length | utf8 JSON header | encoded image bytes

    This provides end-to-end backpressure without queueing obsolete rendered
    tiles and without polling/resending unchanged images.
    """

    route = "/vtkweb/frame-stream"

    def __init__(self, server, *, coalesce_delay: float = 0.003) -> None:
        self.server = server
        self.coalesce_delay = max(0.0, float(coalesce_delay))
        self._clients: set[int] = set()
        self._client_data: dict[int, _Client] = {}
        self._latest_tiles: dict[tuple[str, int], _CachedTile] = {}
        self._revision = 0
        server.controller.on_server_bind.add(self._on_server_bind)

    def _on_server_bind(self, http_server) -> None:
        http_server.app.router.add_get(self.route, self._handle_websocket)

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse(autoping=True, heartbeat=30)
        await websocket.prepare(request)

        key = id(websocket)
        client = _Client(websocket=websocket)
        self._clients.add(key)
        self._client_data[key] = client

        try:
            async for message in websocket:
                if message.type == WSMsgType.TEXT:
                    try:
                        payload = json.loads(message.data)
                    except (TypeError, ValueError):
                        continue
                    if payload.get("type") == "ready":
                        client.credit = True
                        self._schedule_flush(client)
                    continue
                if message.type in {
                    WSMsgType.CLOSE,
                    WSMsgType.CLOSING,
                    WSMsgType.CLOSED,
                }:
                    break
                if message.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(key)
            self._client_data.pop(key, None)
            if client.flush_task is not None and not client.flush_task.done():
                client.flush_task.cancel()
            if not websocket.closed:
                await websocket.close()

        return websocket

    @staticmethod
    def _encode_tile_packet(
        view_id: str,
        image: bytes,
        *,
        mime_type: str,
        generation: int,
        sequence: int,
        region,
        full_size,
        tile_id: int,
        debug: bool,
        size_revision: int,
    ) -> bytes:
        header = json.dumps(
            {
                "view_id": view_id,
                "mime_type": mime_type,
                "generation": int(generation),
                "sequence": int(sequence),
                **(
                    {
                        "region": list(region),
                        "full_size": list(full_size),
                        "tile_id": int(tile_id),
                        "debug": bool(debug),
                        "size_revision": int(size_revision),
                    }
                    if region is not None and full_size is not None
                    else {}
                ),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        return struct.pack(">I", len(header)) + header + image

    @staticmethod
    def _encode_batch(packets: list[bytes]) -> bytes:
        parts = [_BATCH_MAGIC, struct.pack(">I", len(packets))]
        for packet in packets:
            parts.append(struct.pack(">I", len(packet)))
            parts.append(packet)
        return b"".join(parts)

    def _schedule_flush(self, client: _Client) -> None:
        if not client.credit or client.websocket.closed:
            return
        if client.flush_task is not None and not client.flush_task.done():
            return

        async def _coalesced_flush() -> None:
            try:
                if self.coalesce_delay:
                    await asyncio.sleep(self.coalesce_delay)
                await self._flush_client(client)
            except asyncio.CancelledError:
                raise
            finally:
                if client.flush_task is asyncio.current_task():
                    client.flush_task = None

        client.flush_task = asyncio.create_task(_coalesced_flush())

    async def _flush_client(self, client: _Client) -> None:
        if not client.credit or client.websocket.closed:
            return

        async with client.send_lock:
            if not client.credit or client.websocket.closed:
                return

            changed: list[tuple[tuple[str, int], _CachedTile]] = [
                (key, cached)
                for key, cached in self._latest_tiles.items()
                if cached.revision > client.last_sent.get(key, 0)
            ]
            if not changed:
                # Keep the credit outstanding. The next publish will satisfy it.
                return

            # Stable ordering makes captures/debugging deterministic. Tile
            # freshness is still latest-value because there is only one cache
            # entry per logical tile.
            changed.sort(key=lambda item: item[0])
            batch = self._encode_batch([cached.packet for _, cached in changed])

            try:
                await client.websocket.send_bytes(batch)
            except (ConnectionError, RuntimeError):
                return

            for key, cached in changed:
                client.last_sent[key] = cached.revision
            client.credit = False

    def discard_view(self, view_id: str) -> None:
        """Drop cached tiles and per-client revisions for a removed/replaced view."""
        view_id = str(view_id)
        stale_keys = [key for key in self._latest_tiles if key[0] == view_id]
        for key in stale_keys:
            self._latest_tiles.pop(key, None)
            for client in self._client_data.values():
                client.last_sent.pop(key, None)

    async def publish(
        self,
        view_id: str,
        image: bytes,
        *,
        mime_type: str = "image/jpeg",
        generation: int = 0,
        sequence: int = 0,
        region=None,
        full_size=None,
        tile_id: int = 0,
        debug: bool = False,
        size_revision: int = 0,
    ) -> None:
        if not image:
            return

        packet = self._encode_tile_packet(
            view_id,
            image,
            mime_type=mime_type,
            generation=generation,
            sequence=sequence,
            region=region,
            full_size=full_size,
            tile_id=tile_id,
            debug=debug,
            size_revision=size_revision,
        )
        key = (str(view_id), int(tile_id))
        self._revision += 1
        self._latest_tiles[key] = _CachedTile(packet=packet, revision=self._revision)

        # Publishing never queues a frame for a busy client. It only replaces
        # latest state. Clients with an outstanding credit get one short
        # coalescing window so tiles from multiple ranks/views can share a batch.
        for client_key in tuple(self._clients):
            client = self._client_data.get(client_key)
            if client is not None and client.credit and not client.websocket.closed:
                self._schedule_flush(client)
