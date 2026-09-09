from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import dataclass, field
from typing import Any

from aiohttp import WSMsgType, web


@dataclass
class _Client:
    websocket: web.WebSocketResponse
    queue: asyncio.Queue[bytes] = field(
        default_factory=lambda: asyncio.Queue(maxsize=1)
    )
    sender_task: asyncio.Task | None = None


class BinaryFrameTransport:
    """Out-of-state binary frame stream for progressive render backends.

    A frame packet is one websocket binary message:

        uint32_be header_length | utf8 JSON header | encoded image bytes

    Each client owns a single-element queue. If rendering outruns the network,
    the pending frame is replaced so progressive rendering is always
    latest-frame-wins rather than building latency.
    """

    route = "/vtkweb/frame-stream"

    def __init__(self, server) -> None:
        self.server = server
        self._clients: set[int] = set()
        self._client_data: dict[int, _Client] = {}
        server.controller.on_server_bind.add(self._on_server_bind)

    def _on_server_bind(self, http_server) -> None:
        http_server.app.router.add_get(self.route, self._handle_websocket)

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse(autoping=True, heartbeat=30)
        await websocket.prepare(request)

        key = id(websocket)
        client = _Client(websocket=websocket)
        client.sender_task = asyncio.create_task(self._sender(client))
        self._clients.add(key)
        self._client_data[key] = client

        try:
            async for message in websocket:
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
            if client.sender_task is not None:
                client.sender_task.cancel()
                try:
                    await client.sender_task
                except asyncio.CancelledError:
                    pass
            if not websocket.closed:
                await websocket.close()

        return websocket

    async def _sender(self, client: _Client) -> None:
        try:
            while True:
                packet = await client.queue.get()
                await client.websocket.send_bytes(packet)
        except (asyncio.CancelledError, ConnectionError):
            pass

    async def publish(
        self,
        view_id: str,
        image: bytes,
        *,
        mime_type: str = "image/jpeg",
        generation: int = 0,
        sequence: int = 0,
    ) -> None:
        if not image or not self._clients:
            return

        header = json.dumps(
            {
                "view_id": view_id,
                "mime_type": mime_type,
                "generation": int(generation),
                "sequence": int(sequence),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        packet = struct.pack(">I", len(header)) + header + image

        for key in tuple(self._clients):
            client = self._client_data.get(key)
            if client is None or client.websocket.closed:
                continue
            if client.queue.full():
                try:
                    client.queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                client.queue.put_nowait(packet)
            except asyncio.QueueFull:
                pass
