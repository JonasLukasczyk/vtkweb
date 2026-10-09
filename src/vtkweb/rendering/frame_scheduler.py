from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from vtkweb.distributed import TileRegion, context
from vtkweb.rendering.base import FrameRenderingBackend, RenderedFrame
from vtkweb.stream_hub import StreamSink


@dataclass(frozen=True)
class _FrameView:
    backend: FrameRenderingBackend
    backend_view_id: str


class FrameRenderManager:
    """Continuously render RGB tiles and publish each completed tile immediately."""

    def __init__(self, stream_sink: StreamSink | None = None) -> None:
        self.stream_sink = stream_sink
        self._views: dict[str, _FrameView] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._render_sizes: dict[str, tuple[int, int]] = {}
        self._size_revision: dict[str, int] = {}
        self._fps_limit: dict[str, float] = {}
        self._distributed: dict[str, bool] = {}

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

        old_executor = self._executors.pop(view_id, None)
        if old_executor is not None:
            old_executor.shutdown(wait=False, cancel_futures=True)
        self._executors[view_id] = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"vtkweb-render-{view_id[:8]}"
        )
        self.ensure(view_id)

    def unregister_view(self, view_id: str) -> None:
        """Stop only the rank-local renderer for a logical view."""
        frame_view = self._views.pop(view_id, None)
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

    def forget_view(self, view_id: str) -> None:
        self.unregister_view(view_id)
        self._fps_limit.pop(view_id, None)
        self._distributed.pop(view_id, None)
        self._render_sizes.pop(view_id, None)
        self._size_revision.pop(view_id, None)
        if self.stream_sink is not None:
            self.stream_sink.discard_view(view_id)

    def set_fps_limit(self, view_id: str, fps_limit: float) -> None:
        self._fps_limit[view_id] = max(1.0, float(fps_limit))

    def reset_stats(self, view_id: str) -> None:
        if self.stream_sink is None:
            return
        self.stream_sink.reset_stats(view_id)

    def set_distributed(self, view_id: str, distributed: bool) -> None:
        self._distributed[view_id] = bool(distributed and context.enabled)

    def set_render_size(self, view_id: str, width: int, height: int) -> None:
        new_size = (max(1, int(width)), max(1, int(height)))
        old_size = self._render_sizes.get(view_id)
        if old_size != new_size:
            self._size_revision[view_id] = self._size_revision.get(view_id, 0) + 1
            if self.stream_sink is not None:
                self.stream_sink.discard_view(view_id)
        self._render_sizes[view_id] = new_size

    def ensure(self, view_id: str) -> None:
        if view_id not in self._views:
            return
        task = self._tasks.get(view_id)
        if task is None or task.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._tasks[view_id] = loop.create_task(self._render_loop(view_id))

    def ensure_all(self) -> None:
        for view_id in tuple(self._views):
            self.ensure(view_id)

    async def _render_loop(self, view_id: str) -> None:
        try:
            while view_id in self._views:
                frame_view = self._views[view_id]
                backend = frame_view.backend
                backend_view_id = frame_view.backend_view_id

                if not backend.has_renderable_scene(backend_view_id):
                    await asyncio.sleep(0.01)
                    continue

                loop = asyncio.get_running_loop()
                executor = self._executors.get(view_id)
                if executor is None:
                    return

                size = self._render_sizes.get(view_id)
                if size is None:
                    await asyncio.sleep(0.01)
                    continue

                started_at = loop.time()
                size_revision = self._size_revision.get(view_id, 0)
                if self._distributed.get(view_id, False) and context.enabled:
                    tile = context.tile_region(*size)
                else:
                    width, height = size
                    tile = TileRegion(0, 0, width, height, width, height)

                try:
                    frame = await loop.run_in_executor(
                        executor,
                        lambda: backend.render_frame(
                            backend_view_id,
                            region=(tile.x, tile.y, tile.width, tile.height),
                            full_size=(tile.full_width, tile.full_height),
                        ),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    import os
                    if os.environ.get("VTKWEB_VPT_DEBUG", "0").lower() in ("1", "true", "yes"):
                        import traceback
                        print(f"[VPT DEBUG] Frame render failed for {view_id}: {type(exc).__name__}: {exc}", flush=True)
                        traceback.print_exception(type(exc), exc, exc.__traceback__)
                    else:
                        print(f"Frame render failed for {view_id}: {exc}", flush=True)
                    await asyncio.sleep(0.1)
                    continue

                if view_id not in self._views:
                    return

                if frame and self.stream_sink is not None:
                    if not isinstance(frame, RenderedFrame):
                        raise TypeError(
                            f"{type(frame).__name__} returned by renderer; expected RenderedFrame"
                        )
                    try:
                        await self.stream_sink.publish(
                            view_id,
                            frame.rgb,
                            width=frame.width,
                            height=frame.height,
                            region=(tile.x, tile.y, tile.width, tile.height),
                            full_size=(tile.full_width, tile.full_height),
                            size_revision=size_revision,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # Transport/encoding failures should not permanently stop
                        # expensive rendering; report and retry on the next tile.
                        print(
                            f"Frame transport failed for {view_id}: {exc}",
                            flush=True,
                        )

                fps_limit = self._fps_limit.get(view_id, 30.0)
                elapsed = loop.time() - started_at
                await asyncio.sleep(max(0.0, 1.0 / fps_limit - elapsed))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"Frame render loop failed for {view_id}: {exc}", flush=True)
        finally:
            current = asyncio.current_task()
            if self._tasks.get(view_id) is current:
                self._tasks.pop(view_id, None)
