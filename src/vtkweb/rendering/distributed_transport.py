from __future__ import annotations

import asyncio

from vtkweb.distributed import context


class DistributedFrameTransport:
    """Forward raw RGB tiles to rank 0 for immediate CPU composition/encoding."""

    def __init__(self, root_transport=None) -> None:
        self.root_transport = root_transport
        self._receiver_task: asyncio.Task | None = None

    def ensure_receiver(self) -> None:
        if not context.enabled or not context.is_root or self.root_transport is None:
            return
        if self._receiver_task is None or self._receiver_task.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._receiver_task = loop.create_task(self._receive_worker_tiles())

    async def _receive_worker_tiles(self) -> None:
        while True:
            try:
                packet = context.poll_frame()
                if packet is None:
                    await asyncio.sleep(0.001)
                    continue
                for _ in range(32):
                    await self._publish_worker_packet(packet)
                    packet = context.poll_frame()
                    if packet is None:
                        break
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[mpi rank 0] frame receive failed: {exc}", flush=True)
                await asyncio.sleep(0.001)

    async def _publish_worker_packet(self, packet) -> None:
        await self.root_transport.publish(
            packet["view_id"],
            packet["rgb"],
            width=int(packet["width"]),
            height=int(packet["height"]),
            region=tuple(packet["region"]),
            full_size=tuple(packet["full_size"]),
            size_revision=int(packet.get("size_revision", 0)),
            source_rank=int(packet.get("source_rank", 0)),
        )

    def discard_view(self, view_id: str) -> None:
        if self.root_transport is not None:
            self.root_transport.discard_view(view_id)

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

        if not context.enabled:
            if self.root_transport is not None:
                await self.root_transport.publish(view_id, rgb, **kwargs)
            return

        if context.is_root:
            self.ensure_receiver()
            if self.root_transport is not None:
                await self.root_transport.publish(view_id, rgb, **kwargs)
            return

        context.send_frame({"view_id": view_id, "rgb": rgb, **kwargs})
