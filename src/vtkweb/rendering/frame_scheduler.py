from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from vtkweb.rendering.base import FrameRenderingBackend
from vtkweb.rendering.frame_transport import FrameTransport


@dataclass(frozen=True)
class _FrameView:
    backend: FrameRenderingBackend
    backend_view_id: str


class FrameRenderManager:
    """Continuously render server-side views and publish encoded frames.

    Each logical view owns one single-thread executor so graphics contexts remain
    thread-affine. The scheduler does not snapshot or discard scene generations:
    every frame returned by the backend is published. Backend state changes are
    simply observed by the next render-loop iteration.
    """

    def __init__(self, frame_transport: FrameTransport | None = None) -> None:
        self.frame_transport = frame_transport
        self._views: dict[str, _FrameView] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._frame_sequence: dict[str, int] = {}
        # Transport epoch only. It prevents a frame from a previous backend/view
        # incarnation from replacing frames after a backend switch.
        self._stream_generation: dict[str, int] = {}
        self._executors: dict[str, ThreadPoolExecutor] = {}

    def register_view(
        self,
        view_id: str,
        backend: FrameRenderingBackend,
        backend_view_id: str,
    ) -> None:
        old_task = self._tasks.pop(view_id, None)
        if old_task is not None and not old_task.done():
            old_task.cancel()

        self._views[view_id] = _FrameView(backend, backend_view_id)
        self._stream_generation[view_id] = self._stream_generation.get(view_id, 0) + 1

        old_executor = self._executors.pop(view_id, None)
        if old_executor is not None:
            old_executor.shutdown(wait=False, cancel_futures=True)
        self._executors[view_id] = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"vtkweb-render-{view_id[:8]}"
        )
        self._frame_sequence[view_id] = 0
        self.ensure(view_id)

    def unregister_view(self, view_id: str) -> None:
        frame_view = self._views.pop(view_id, None)
        self._frame_sequence.pop(view_id, None)
        task = self._tasks.pop(view_id, None)
        if task is not None and not task.done():
            task.cancel()
        executor = self._executors.pop(view_id, None)
        if executor is not None:
            cleanup = getattr(
                frame_view.backend if frame_view is not None else None,
                "release_render_resources",
                None,
            )
            if callable(cleanup) and frame_view is not None:
                try:
                    executor.submit(cleanup, frame_view.backend_view_id).result()
                except Exception:
                    pass
            executor.shutdown(wait=True, cancel_futures=True)

    def ensure(self, view_id: str) -> None:
        if view_id not in self._views:
            return
        task = self._tasks.get(view_id)
        if task is None or task.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Views can be created during module import before Trame starts
                # its event loop. A later resize/state event calls ensure again.
                return
            stream_generation = self._stream_generation.get(view_id, 0)
            self._tasks[view_id] = loop.create_task(
                self._render_loop(view_id, stream_generation)
            )

    def ensure_all(self) -> None:
        for view_id in tuple(self._views):
            self.ensure(view_id)

    async def _render_loop(self, view_id: str, stream_generation: int) -> None:
        try:
            while (
                view_id in self._views
                and self._stream_generation.get(view_id) == stream_generation
            ):
                frame_view = self._views[view_id]
                backend = frame_view.backend
                backend_view_id = frame_view.backend_view_id

                if not backend.has_renderable_scene(backend_view_id):
                    return

                loop = asyncio.get_running_loop()
                executor = self._executors.get(view_id)
                if executor is None:
                    return

                started_at = loop.time()
                frame = await loop.run_in_executor(
                    executor, backend.render_frame, backend_view_id
                )

                if (
                    view_id not in self._views
                    or self._stream_generation.get(view_id) != stream_generation
                ):
                    return

                if frame and self.frame_transport is not None:
                    sequence = self._frame_sequence.get(view_id, 0) + 1
                    self._frame_sequence[view_id] = sequence
                    await self.frame_transport.publish(
                        view_id,
                        frame,
                        generation=stream_generation,
                        sequence=sequence,
                    )

                # Yield to Trame/network handling between completed frames.
                # Fast raster backends may optionally cap their continuous loop
                # so they do not monopolize CPU/GPU resources needed by other
                # views. Progressive backends simply omit ``target_fps``.
                target_fps = getattr(backend, "target_fps", None)
                if target_fps:
                    elapsed = loop.time() - started_at
                    await asyncio.sleep(max(0.0, 1.0 / float(target_fps) - elapsed))
                else:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"Frame render loop failed for {view_id}: {exc}")
        finally:
            current = asyncio.current_task()
            if self._tasks.get(view_id) is current:
                self._tasks.pop(view_id, None)
