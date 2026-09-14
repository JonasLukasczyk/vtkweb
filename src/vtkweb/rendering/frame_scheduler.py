from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from vtkweb.rendering.base import FrameRenderingBackend
from vtkweb.rendering.frame_transport import FrameTransport
from vtkweb.distributed import TileRegion, context


@dataclass(frozen=True)
class _FrameView:
    backend: FrameRenderingBackend
    backend_view_id: str


class FrameRenderManager:
    """Continuously render server-side views and publish encoded frames.

    Each logical view owns one single-thread executor so graphics contexts remain
    thread-affine. Every completed frame is offered to the transport; the
    browser-facing transport is latest-value/credit-driven, so superseded tiles
    may be replaced before they ever cross the websocket. Backend state changes
    are simply observed by the next render-loop iteration.
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
        self._render_sizes: dict[str, tuple[int, int]] = {}
        self._size_revision: dict[str, int] = {}
        self._debug: dict[str, bool] = {}
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
        set_transport_mode = getattr(self.frame_transport, "set_view_distributed", None)
        if callable(set_transport_mode):
            set_transport_mode(view_id, False)
        else:
            discard = getattr(self.frame_transport, "discard_view", None)
            if callable(discard):
                discard(view_id)
        self._frame_sequence.pop(view_id, None)
        self._fps_limit.pop(view_id, None)
        self._distributed.pop(view_id, None)
        self._render_sizes.pop(view_id, None)
        self._size_revision.pop(view_id, None)
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

    def set_debug(self, view_id: str, enabled: bool) -> None:
        self._debug[view_id] = bool(enabled)

    def set_fps_limit(self, view_id: str, fps_limit: float) -> None:
        """Set the continuous-rendering limit for one logical view on this rank."""
        self._fps_limit[view_id] = max(1.0, float(fps_limit))

    def set_distributed(self, view_id: str, distributed: bool) -> None:
        """Select full-frame root rendering or MPI tile rendering for a view."""
        distributed = bool(distributed and context.enabled)
        previous = self._distributed.get(view_id, False)
        self._distributed[view_id] = distributed
        set_transport_mode = getattr(self.frame_transport, "set_view_distributed", None)
        if callable(set_transport_mode):
            set_transport_mode(view_id, distributed)
        if previous == distributed:
            return
        discard = getattr(self.frame_transport, "discard_view", None)
        if callable(discard):
            discard(view_id)
        if view_id in self._views:
            self._stream_generation[view_id] = (
                self._stream_generation.get(view_id, 0) + 1
            )
            task = self._tasks.pop(view_id, None)
            if task is not None and not task.done():
                task.cancel()
            self.ensure(view_id)

    def set_render_size(
        self,
        view_id: str,
        width: int,
        height: int,
        *,
        revision: int | None = None,
    ) -> None:
        new_size = (max(1, int(width)), max(1, int(height)))
        old_size = self._render_sizes.get(view_id)
        if old_size != new_size:
            if revision is None:
                revision = self._size_revision.get(view_id, 0) + 1
            self._size_revision[view_id] = max(1, int(revision))
            discard = getattr(self.frame_transport, "discard_view", None)
            if callable(discard):
                discard(view_id)
        self._render_sizes[view_id] = new_size

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

                # Dynamically-created views can be registered before the browser
                # has reported their viewport size (and before backend runtime
                # state is fully ready). Keep the per-view loop alive instead of
                # exiting permanently; later replicated size/state mutations can
                # make the view renderable without needing a fragile restart race.
                if not backend.has_renderable_scene(backend_view_id):
                    await asyncio.sleep(0.01)
                    continue

                loop = asyncio.get_running_loop()
                executor = self._executors.get(view_id)
                if executor is None:
                    return

                started_at = loop.time()
                size = self._render_sizes.get(view_id)
                if size is None:
                    # A newly opened render view exists on every MPI rank before
                    # its ResizeObserver reports dimensions through rank 0. Stay
                    # registered and wait for that replicated size update.
                    await asyncio.sleep(0.01)
                    continue
                size_revision = self._size_revision.get(view_id, 0)
                if self._distributed.get(view_id, False) and context.enabled:
                    tile = context.tile_region(*size)
                else:
                    width, height = size
                    tile = TileRegion(0, 0, width, height, width, height, 0)
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
                    # A transient backend failure (notably Dr.Jit/Mitsuba) must
                    # not permanently kill this rank/view render task. Lose one
                    # frame, back off briefly, and let the next iteration retry.
                    print(
                        f"Frame render failed for {view_id}: {exc}",
                        flush=True,
                    )
                    await asyncio.sleep(0.1)
                    continue

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
                        region=(tile.x, tile.y, tile.width, tile.height),
                        full_size=(tile.full_width, tile.full_height),
                        tile_id=tile.tile_id,
                        debug=self._debug.get(view_id, False),
                        size_revision=size_revision,
                    )

                # Every backend uses the same per-view FPS limit. The limit is
                # per rank, so N MPI ranks may collectively produce up to
                # N * fps_limit tile frames for one logical view.
                fps_limit = self._fps_limit.get(view_id, 30.0)
                elapsed = loop.time() - started_at
                await asyncio.sleep(max(0.0, 1.0 / fps_limit - elapsed))
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"Frame render loop failed for {view_id}: {exc}")
        finally:
            current = asyncio.current_task()
            if self._tasks.get(view_id) is current:
                self._tasks.pop(view_id, None)
