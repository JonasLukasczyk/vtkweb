from __future__ import annotations

import asyncio
from datetime import datetime
from time import perf_counter

import vtk

from vtkweb.pipeline import PipelineGraph


class PipelineExecutionManager:
    """Explicit, application-level scheduler for the VTK pipeline graph."""

    def __init__(self, state, pipeline: PipelineGraph, rendering) -> None:
        self.state = state
        self.pipeline = pipeline
        self.rendering = rendering
        self._abort_requested = False
        self._running = False
        self.state.pipeline_executing = False

    @property
    def running(self) -> bool:
        return self._running

    def abort(self) -> None:
        if self._running:
            self._abort_requested = True

    async def execute(self) -> None:
        if self._running:
            return

        self._running = True
        self._abort_requested = False
        self.state.pipeline_executing = True

        try:
            if not self.pipeline.modified_node_ids():
                active = self.pipeline.active_node_id
                if active is None:
                    return
                self.pipeline.mark_modified(active, include_downstream=False)

            while True:
                order = self.pipeline.topological_order()
                self._log_schedule("topological", order)
                modified = next(
                    (
                        node_id
                        for node_id in order
                        if self.pipeline.execution_state(node_id) == "modified"
                    ),
                    None,
                )
                if modified is None:
                    break

                subgraph = self.pipeline.downstream_subgraph(modified)
                scheduled = [node_id for node_id in order if node_id in subgraph]
                self._log_schedule(
                    f"subgraph from {self._node_label(modified)}", scheduled
                )
                self.pipeline.set_execution_states(scheduled, "queued")
                self._flush_state()

                for node_id in scheduled:
                    if self._abort_requested:
                        self._fail_all_queued()
                        return

                    # A node can have been re-modified while an earlier node ran.
                    if self.pipeline.execution_state(node_id) not in {
                        "queued",
                        "modified",
                    }:
                        continue

                    self.pipeline.set_execution_state(node_id, "running")
                    self._flush_state()

                    # Deliberately leave the node in its freshly-pushed
                    # ``running`` state for a moment before entering VTK.
                    # This makes scheduler transitions easy to observe in the
                    # pipeline browser while debugging.
                    await asyncio.sleep(0.5)

                    started_at = datetime.now().astimezone()
                    started_perf = perf_counter()
                    print(
                        f"[execution] START {self._node_label(node_id)} "
                        f"at {started_at.isoformat(timespec='milliseconds')}",
                        flush=True,
                    )
                    modification_version = self.pipeline.modification_version(node_id)
                    errors: list[str] = []
                    processor = self.pipeline.processor(node_id)

                    def on_error(_obj, _event, message=None):
                        errors.append(str(message or "VTK execution error"))

                    observer_id = processor.AddObserver(
                        vtk.vtkCommand.ErrorEvent, on_error
                    )
                    try:
                        self._resolve_inputs(node_id)
                        await asyncio.to_thread(processor.Update)
                    except Exception as exc:
                        errors.append(str(exc))
                    finally:
                        processor.RemoveObserver(observer_id)

                    elapsed = perf_counter() - started_perf
                    ended_at = datetime.now().astimezone()

                    if errors:
                        self.pipeline.set_execution_state(node_id, "failed")
                        self._flush_state()
                        print(
                            f"[execution] END   {self._node_label(node_id)} "
                            f"at {ended_at.isoformat(timespec='milliseconds')} "
                            f"after {elapsed:.3f}s -> failed: {'; '.join(errors)}",
                            flush=True,
                        )
                        self._fail_queued_downstream(node_id)
                        return

                    # If a property/input changed while this node was executing,
                    # keep it modified so the outer loop will pick it up again.
                    if (
                        self.pipeline.modification_version(node_id)
                        != modification_version
                    ):
                        self.pipeline.set_execution_state(node_id, "modified")
                    else:
                        self.pipeline.set_execution_state(node_id, "success")

                    final_state = self.pipeline.execution_state(node_id)
                    print(
                        f"[execution] END   {self._node_label(node_id)} "
                        f"at {ended_at.isoformat(timespec='milliseconds')} "
                        f"after {elapsed:.3f}s -> {final_state}",
                        flush=True,
                    )

                    # Refresh downstream concrete inputs to point at this
                    # node's latest output objects. This is wiring only; it does
                    # not execute downstream algorithms.
                    if final_state == "success":
                        self.pipeline.bind_downstream_inputs(node_id)

                    # Refresh UI metadata and visual representations from the
                    # already-computed output. These calls must not execute the
                    # computational pipeline. Missing output representations are
                    # created only after the node has succeeded at least once.
                    self.pipeline.sync_node_from_runtime(node_id)
                    if final_state == "success":
                        self.rendering.ensure_output_representations(node_id)
                    self.rendering.refresh_node(node_id)
                    self._flush_state()

                    # Keep the completed state visible before scheduling the
                    # next node. This is intentionally outside the measured VTK
                    # execution duration printed above.
                    await asyncio.sleep(0.5)

                # Required execution model: graph mutations performed during an
                # execution are respected by sorting and scanning from the start.

        except Exception:
            queued = [
                node_id
                for node_id in self.pipeline.nodes
                if self.pipeline.execution_state(node_id) == "queued"
            ]
            self.pipeline.set_execution_states(queued, "failed")
            self._flush_state()
            raise
        finally:
            self._running = False
            self._abort_requested = False
            self.state.pipeline_executing = False
            self._flush_state()

    def _flush_state(self) -> None:
        """Push async scheduler state changes to Trame immediately."""
        self.state.flush()

    def _node_label(self, node_id: str) -> str:
        """Return a stable, human-readable node label for debug output."""
        try:
            node = self.pipeline.node_state(node_id)
            name = node.get("name") or node.get("class_name") or "node"
            return f"{name} ({node_id})"
        except Exception:
            return str(node_id)

    def _log_schedule(self, label: str, node_ids) -> None:
        nodes = " -> ".join(self._node_label(node_id) for node_id in node_ids)
        print(f"[execution] schedule [{label}]: {nodes or '<empty>'}", flush=True)

    def _resolve_inputs(self, target_node_id: str) -> None:
        # Re-materialize model edges immediately before execution as a final
        # consistency check. PipelineGraph.bind_inputs() performs no Update()
        # or UpdateInformation() and introduces no data copies.
        self.pipeline.bind_inputs(target_node_id)

    def _fail_all_queued(self) -> None:
        failed = [
            item
            for item in self.pipeline.nodes
            if self.pipeline.execution_state(item) == "queued"
        ]
        self.pipeline.set_execution_states(failed, "failed")
        self._flush_state()

    def _fail_queued_downstream(self, node_id: str) -> None:
        downstream = self.pipeline.downstream_subgraph(node_id)
        failed = [
            item
            for item in downstream
            if self.pipeline.execution_state(item) == "queued"
        ]
        self.pipeline.set_execution_states(failed, "failed")
        self._flush_state()
