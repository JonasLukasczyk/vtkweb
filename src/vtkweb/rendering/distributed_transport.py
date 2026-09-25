from __future__ import annotations

import asyncio

from vtkweb.distributed import context


class DistributedFrameTransport:
    """Forward encoded per-rank tiles through rank 0 to the browser transport."""

    def __init__(self, root_transport=None) -> None:
        self.root_transport = root_transport
        self._receiver_task: asyncio.Task | None = None
        self._distributed_views: set[str] = set()

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

                # Drain a bounded burst before yielding. With many ranks, frame
                # completions often arrive together; handling several queued
                # packets per event-loop turn avoids needless asyncio wakeups.
                for _ in range(32):
                    await self._publish_worker_packet(packet)
                    packet = context.poll_frame()
                    if packet is None:
                        break
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Keep the long-lived receiver alive so one malformed/failed
                # packet cannot create an endless sequence of orphaned tasks.
                print(f"[mpi rank 0] frame receive failed: {exc}", flush=True)
                await asyncio.sleep(0.001)

    async def _publish_worker_packet(self, packet) -> None:
        if str(packet["view_id"]) not in self._distributed_views:
            # A view may have just switched back to rank-0-only rendering while
            # worker tiles were still in flight. Drop them so stale MPI tiles
            # cannot repopulate the browser cache.
            return
        await self.root_transport.publish(
            packet["view_id"],
            packet["image"],
            mime_type=packet.get("mime_type", "image/jpeg"),
            generation=packet.get("generation", 0),
            sequence=packet.get("sequence", 0),
            region=tuple(packet["region"]),
            full_size=tuple(packet["full_size"]),
            tile_id=packet.get("tile_id", packet.get("rank", 0)),
            debug=packet.get("debug", False),
            size_revision=packet.get("size_revision", 0),
            depth=packet.get("depth"),
            depth_near=packet.get("depth_near"),
            depth_far=packet.get("depth_far"),
            depth_encoding=packet.get("depth_encoding"),
            ssao_slices=int(packet.get("ssao_slices", 0)),
            ssao_steps=int(packet.get("ssao_steps", 6)),
            ssao_radius=float(packet.get("ssao_radius", 10.0)),
            ssao_strength=float(packet.get("ssao_strength", 1.0)),
            ssao_thickness=float(packet.get("ssao_thickness", 0.4)),
            camera_fov=float(packet.get("camera_fov", 30.0)),
        )

    def set_view_distributed(self, view_id: str, enabled: bool) -> None:
        view_id = str(view_id)
        if enabled:
            self._distributed_views.add(view_id)
        else:
            self._distributed_views.discard(view_id)
        self.discard_view(view_id)

    def discard_view(self, view_id: str) -> None:
        if self.root_transport is None:
            return
        discard = getattr(self.root_transport, "discard_view", None)
        if callable(discard):
            discard(view_id)

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
        depth: bytes | None = None,
        depth_near: float | None = None,
        depth_far: float | None = None,
        depth_encoding: str | None = None,
        ssao_slices: int = 0,
        ssao_steps: int = 6,
        ssao_radius: float = 10.0,
        ssao_strength: float = 1.0,
        ssao_thickness: float = 0.4,
        camera_fov: float = 30.0,
    ) -> None:
        if not image:
            return
        if not context.enabled:
            if self.root_transport is not None:
                await self.root_transport.publish(
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
                    depth=depth,
                    depth_near=depth_near,
                    depth_far=depth_far,
                    depth_encoding=depth_encoding,
                    ssao_slices=ssao_slices,
                    ssao_steps=ssao_steps,
                    ssao_radius=ssao_radius,
                    ssao_strength=ssao_strength,
                    ssao_thickness=ssao_thickness,
                    camera_fov=camera_fov,
                )
            return

        if context.is_root:
            self.ensure_receiver()
            if self.root_transport is not None:
                await self.root_transport.publish(
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
                    depth=depth,
                    depth_near=depth_near,
                    depth_far=depth_far,
                    depth_encoding=depth_encoding,
                    ssao_slices=ssao_slices,
                    ssao_steps=ssao_steps,
                    ssao_radius=ssao_radius,
                    ssao_strength=ssao_strength,
                    ssao_thickness=ssao_thickness,
                    camera_fov=camera_fov,
                )
            return

        context.send_frame(
            {
                "view_id": view_id,
                "image": image,
                "mime_type": mime_type,
                "generation": int(generation),
                "sequence": int(sequence),
                "region": tuple(region),
                "full_size": tuple(full_size),
                "tile_id": int(tile_id),
                "debug": bool(debug),
                "size_revision": int(size_revision),
                "depth": depth,
                "depth_near": depth_near,
                "depth_far": depth_far,
                "depth_encoding": depth_encoding,
                "ssao_slices": int(ssao_slices),
                "ssao_steps": int(ssao_steps),
                "ssao_radius": float(ssao_radius),
                "ssao_strength": float(ssao_strength),
                "ssao_thickness": float(ssao_thickness),
                "camera_fov": float(camera_fov),
            }
        )
