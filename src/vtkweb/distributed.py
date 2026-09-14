from __future__ import annotations

import asyncio
import atexit
import inspect
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable

COMMAND_TAG = 41001
FRAME_TAG = 41002
STOP_TAG = 41003


@dataclass(frozen=True)
class TileRegion:
    x: int
    y: int
    width: int
    height: int
    full_width: int
    full_height: int
    tile_id: int


class DistributedContext:
    """Small optional MPI facade used by vtkweb.

    MPI is intentionally runtime infrastructure only.  When mpi4py is absent or
    COMM_WORLD has one rank this object reduces to the single-process behavior.
    """

    def __init__(self) -> None:
        self._mpi = None
        self.mpi_comm = None
        self.vtk_controller = None
        self.rank = 0
        self.size = 1
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._pending = []
        # Keep nonblocking frame-send requests (and their packet objects) alive
        # until MPI reports completion. Dropping an mpi4py object-mode isend
        # request early can invalidate the serialized send buffer under load.
        self._frame_pending: list[tuple[Any, dict[str, Any]]] = []
        self._mpi_lock = threading.Lock()
        try:
            from mpi4py import MPI
        except ImportError:
            return

        self._mpi = MPI
        self.mpi_comm = MPI.COMM_WORLD
        self.rank = int(self.mpi_comm.Get_rank())
        self.size = int(self.mpi_comm.Get_size())

        # Expose a VTK controller when VTK was built with MPI support.  It is
        # never serialized into application state.
        if self.size > 1:
            try:
                import vtk

                controller = vtk.vtkMPIController()
                # mpi4py already initialized MPI; do not let VTK own finalize.
                controller.Initialize(None, None, 1)
                self.vtk_controller = controller
            except Exception:
                self.vtk_controller = None

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.size > 1

    def register(self, operation: str, handler: Callable[..., Any]) -> None:
        self._handlers[str(operation)] = handler

    def replicate(self, operation: str, *args, **kwargs) -> None:
        """Asynchronously enqueue one root-originated mutation on all workers."""
        if not self.enabled or not self.is_root:
            return
        packet = {"operation": str(operation), "args": args, "kwargs": kwargs}
        with self._mpi_lock:
            self._pending = [request for request in self._pending if not request.Test()]
            for destination in range(1, self.size):
                self._pending.append(
                    self.mpi_comm.isend(packet, dest=destination, tag=COMMAND_TAG)
                )

    async def worker_loop(self) -> None:
        """Run the non-root command pump without blocking render coroutines."""
        if not self.enabled or self.is_root:
            return
        while True:
            packet = None
            with self._mpi_lock:
                if self.mpi_comm.Iprobe(source=0, tag=STOP_TAG):
                    self.mpi_comm.recv(source=0, tag=STOP_TAG)
                    return
                if self.mpi_comm.Iprobe(source=0, tag=COMMAND_TAG):
                    packet = self.mpi_comm.recv(source=0, tag=COMMAND_TAG)
            if packet is None:
                await asyncio.sleep(0.001)
                continue

            operation = packet["operation"]
            handler = self._handlers.get(operation)
            if handler is None:
                print(
                    f"[mpi rank {self.rank}] unknown operation: {operation}", flush=True
                )
                continue
            try:
                result = handler(*packet.get("args", ()), **packet.get("kwargs", {}))
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                print(
                    f"[mpi rank {self.rank}] operation {operation!r} failed: {exc}",
                    flush=True,
                )

    def tile_region(self, width: int, height: int) -> TileRegion:
        """Return a deterministic near-square image-space tile for this rank."""
        width = max(1, int(width))
        height = max(1, int(height))
        if not self.enabled:
            return TileRegion(0, 0, width, height, width, height, 0)

        columns = int(math.ceil(math.sqrt(self.size)))
        rows = int(math.ceil(self.size / columns))
        column = self.rank % columns
        row = self.rank // columns

        x0 = (width * column) // columns
        x1 = (width * (column + 1)) // columns
        y0 = (height * row) // rows
        y1 = (height * (row + 1)) // rows
        return TileRegion(
            x0, y0, max(1, x1 - x0), max(1, y1 - y0), width, height, self.rank
        )

    def execution_success(self, local_ok: bool, local_error: str | None = None):
        """Collectively synchronize one pipeline node and propagate failures."""
        if not self.enabled:
            return bool(local_ok), [local_error] if local_error else []
        with self._mpi_lock:
            all_ok = bool(self.mpi_comm.allreduce(bool(local_ok), op=self._mpi.LAND))
            errors = self.mpi_comm.gather(local_error, root=0)
        if self.is_root:
            return all_ok, [item for item in errors if item]
        return all_ok, []

    def send_frame(self, packet: dict[str, Any]) -> None:
        if not self.enabled or self.is_root:
            return
        with self._mpi_lock:
            # Reap completed sends, but retain every outstanding Request and
            # packet until completion. This is essential for mpi4py object-mode
            # nonblocking sends, especially during high-frequency camera motion.
            self._frame_pending = [
                (request, pending_packet)
                for request, pending_packet in self._frame_pending
                if not request.Test()
            ]
            # Latest-value backpressure begins at the worker as well: never
            # queue multiple encoded frames for the same logical view while an
            # older tile is still in flight to rank 0. Continuous render loops
            # will naturally offer a newer frame after the send completes.
            view_id = str(packet.get("view_id", ""))
            if any(
                str(pending_packet.get("view_id", "")) == view_id
                for _request, pending_packet in self._frame_pending
            ):
                return
            request = self.mpi_comm.isend(packet, dest=0, tag=FRAME_TAG)
            self._frame_pending.append((request, packet))

    def shutdown_workers(self) -> None:
        if not self.enabled or not self.is_root:
            return
        try:
            with self._mpi_lock:
                for destination in range(1, self.size):
                    self.mpi_comm.isend(None, dest=destination, tag=STOP_TAG)
        except Exception:
            pass

    def poll_frame(self) -> dict[str, Any] | None:
        if not self.enabled or not self.is_root:
            return None
        with self._mpi_lock:
            if not self.mpi_comm.Iprobe(source=self._mpi.ANY_SOURCE, tag=FRAME_TAG):
                return None
            status = self._mpi.Status()
            packet = self.mpi_comm.recv(
                source=self._mpi.ANY_SOURCE, tag=FRAME_TAG, status=status
            )
        packet.setdefault("rank", int(status.Get_source()))
        return packet


context = DistributedContext()
atexit.register(context.shutdown_workers)
