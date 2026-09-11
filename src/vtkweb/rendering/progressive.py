from __future__ import annotations

import asyncio
from dataclasses import dataclass

from vtkweb.rendering.base import ProgressiveRenderingBackend
from vtkweb.rendering.frame_transport import FrameTransport


@dataclass(frozen=True)
class _ProgressiveView:
    backend: ProgressiveRenderingBackend
    backend_view_id: str


class ProgressiveRenderManager:
    """Own progressive render tasks and out-of-state frame publication."""

    def __init__(self, frame_transport: FrameTransport | None = None) -> None:
        self.frame_transport = frame_transport
        self._views: dict[str, _ProgressiveView] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._frame_sequence: dict[str, int] = {}

    def register_view(
        self,
        view_id: str,
        backend: ProgressiveRenderingBackend,
        backend_view_id: str,
    ) -> None:
        self._views[view_id] = _ProgressiveView(backend, backend_view_id)
        self.ensure(view_id)

    def unregister_view(self, view_id: str) -> None:
        self._views.pop(view_id, None)
        self._frame_sequence.pop(view_id, None)
        task = self._tasks.pop(view_id, None)
        if task is not None and not task.done():
            task.cancel()

    def ensure(self, view_id: str) -> None:
        if view_id not in self._views:
            return
        task = self._tasks.get(view_id)
        if task is None or task.done():
            self._tasks[view_id] = asyncio.create_task(self._render_loop(view_id))

    def ensure_all(self) -> None:
        for view_id in tuple(self._views):
            self.ensure(view_id)

    async def _render_loop(self, view_id: str) -> None:
        try:
            while view_id in self._views:
                progressive_view = self._views[view_id]
                backend = progressive_view.backend
                backend_view_id = progressive_view.backend_view_id

                if not backend.has_renderable_scene(backend_view_id):
                    await asyncio.sleep(0.05)
                    continue

                generation, camera = backend.render_snapshot(backend_view_id)
                if backend.accumulation_generation(backend_view_id) != generation:
                    backend.clear_accumulation(backend_view_id, generation)

                sample = await asyncio.to_thread(
                    backend.render_pass,
                    backend_view_id,
                    camera,
                    spp=1,
                )

                # A view can be removed while the blocking render runs in a
                # worker thread. Do not touch its backend handle afterward.
                if view_id not in self._views:
                    return

                current_generation, _ = backend.render_snapshot(backend_view_id)
                if current_generation != generation:
                    # The scene changed while this pass was in flight. It may
                    # still be displayed, but it must never contaminate the new
                    # progressive accumulation buffer.
                    backend.clear_accumulation(backend_view_id, current_generation)
                    frame = await asyncio.to_thread(backend.encoded_frame, sample)
                else:
                    backend.accumulate_pass(
                        backend_view_id,
                        sample,
                        spp=1,
                        generation=generation,
                    )
                    frame = await asyncio.to_thread(
                        backend.encoded_accumulated_frame,
                        backend_view_id,
                    )

                if frame is not None and self.frame_transport is not None:
                    sequence = self._frame_sequence.get(view_id, 0) + 1
                    self._frame_sequence[view_id] = sequence
                    await self.frame_transport.publish(
                        view_id,
                        frame,
                        generation=current_generation,
                        sequence=sequence,
                    )

                await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"Progressive render loop failed for {view_id}: {exc}")
        finally:
            current = asyncio.current_task()
            if self._tasks.get(view_id) is current:
                self._tasks.pop(view_id, None)
