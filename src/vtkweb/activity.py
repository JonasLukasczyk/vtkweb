from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from vtkweb.distributed import context


@dataclass
class _RankActivity:
    generation: int = 0
    progress: float | None = None
    details: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    duration: float | None = None
    done: bool = False


@dataclass
class _Activity:
    key: str
    view_id: str
    label: str
    determinate: bool
    ranks: dict[int, _RankActivity] = field(default_factory=dict)
    first_started_at: float = 0.0
    completed_at: float | None = None


class DistributedActivityReporter:
    """Low-rate progress reporting for expensive distributed rendering work."""

    def __init__(self, server=None, publisher=None) -> None:
        self.server = server if context.is_root else None
        self._publisher = publisher if context.is_root else None
        self._lock = threading.Lock()
        self._local: deque[dict[str, Any]] = deque()
        self._activities: dict[str, _Activity] = {}
        self._generations: dict[str, int] = {}
        self._receiver_task: asyncio.Task | None = None

    def start(
        self,
        key: str,
        label: str,
        *,
        view_id: str,
        details: dict[str, Any] | None = None,
        determinate: bool = True,
    ) -> float:
        started_at = time.monotonic()
        key = str(key)
        generation = self._generations.get(key, 0) + 1
        self._generations[key] = generation
        self._emit(
            {
                "kind": "start",
                "key": key,
                "generation": generation,
                "view_id": str(view_id),
                "label": str(label),
                "determinate": bool(determinate),
                "details": dict(details or {}),
                "started_at": started_at,
                "rank": context.rank,
            }
        )
        return started_at

    def progress(self, key: str, progress: float) -> None:
        self._emit(
            {
                "kind": "progress",
                "key": str(key),
                "generation": self._generations.get(str(key), 0),
                "progress": max(0.0, min(1.0, float(progress))),
                "rank": context.rank,
            }
        )

    def done(self, key: str, started_at: float, *, details: dict[str, Any] | None = None) -> None:
        self._emit(
            {
                "kind": "done",
                "key": str(key),
                "generation": self._generations.get(str(key), 0),
                "duration": max(0.0, time.monotonic() - float(started_at)),
                "details": dict(details or {}),
                "rank": context.rank,
            }
        )

    def _emit(self, event: dict[str, Any]) -> None:
        if context.is_root:
            with self._lock:
                self._local.append(event)
        else:
            context.send_activity(event)


    def ensure_receiver(self) -> None:
        """Start an independent low-rate activity drain on rank 0."""
        if not context.is_root:
            return
        if self._receiver_task is not None and not self._receiver_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._receiver_task = loop.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        while True:
            try:
                changed = self.poll()
                await asyncio.sleep(0 if changed else 0.01)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[activity] receive failed: {exc}", flush=True)
                await asyncio.sleep(0.05)

    def poll(self) -> bool:
        """Drain local/MPI events on rank 0. Returns True when UI state changed."""
        if not context.is_root:
            return False
        changed = False
        while True:
            with self._lock:
                event = self._local.popleft() if self._local else None
            if event is None:
                break
            self._apply(event)
            changed = True
        while True:
            event = context.poll_activity()
            if event is None:
                break
            self._apply(event)
            changed = True
        if changed:
            self._publish()
        return changed

    def _apply(self, event: dict[str, Any]) -> None:
        key = str(event["key"])
        generation = int(event.get("generation", 0))
        rank = int(event.get("rank", 0))
        kind = str(event.get("kind", ""))
        activity = self._activities.get(key)

        # Generation counters are local to each MPI rank. A newer generation
        # supersedes only older events from that same rank.
        if kind == "start":
            if activity is None:
                activity = _Activity(
                    key=key,
                    view_id=str(event.get("view_id", "")),
                    label=str(event.get("label", key)),
                    determinate=bool(event.get("determinate", True)),
                )
                self._activities[key] = activity
            current = activity.ranks.get(rank)
            if current is not None and generation < current.generation:
                return
            received_at = time.monotonic()
            if activity.first_started_at <= 0.0 or activity.completed_at is not None:
                activity.first_started_at = received_at
            activity.ranks[rank] = _RankActivity(
                generation=generation,
                progress=0.0 if activity.determinate else None,
                details=dict(event.get("details") or {}),
                started_at=received_at,
            )
            activity.completed_at = None
            return

        if activity is None:
            return
        rank_state = activity.ranks.get(rank)
        if rank_state is None or generation != rank_state.generation:
            return
        if kind == "progress":
            rank_state.progress = float(event.get("progress", 0.0))
        elif kind == "done":
            rank_state.progress = 1.0
            rank_state.duration = float(event.get("duration", 0.0))
            rank_state.done = True
            rank_state.details.update(event.get("details") or {})
            if all(
                activity.ranks.get(index) is not None
                and activity.ranks[index].done
                for index in range(context.size)
            ):
                activity.completed_at = time.monotonic()

    @staticmethod
    def _compact_value(value: Any) -> str:
        if isinstance(value, (tuple, list)):
            values = [int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v for v in value]
            if len(values) == 3 and values[0] == values[1] == values[2]:
                return f"{values[0]}³"
            return "×".join(str(v) for v in values)
        return str(value)

    def _detail_lines(self, activity: _Activity) -> list[str]:
        names: list[str] = []
        for rank_state in activity.ranks.values():
            for name in rank_state.details:
                if name not in names:
                    names.append(name)
        lines = []
        for name in names:
            values = []
            present = []
            for rank in range(context.size):
                rank_state = activity.ranks.get(rank)
                if rank_state is None or name not in rank_state.details:
                    values.append("—")
                else:
                    value = self._compact_value(rank_state.details[name])
                    values.append(value)
                    present.append(value)
            if present and len(present) == context.size and len(set(present)) == 1:
                lines.append(f"{name}={present[0]}")
            else:
                lines.append(f"{name}=[{'|'.join(values)}]")
        return lines

    def _publish(self) -> None:
        if self.server is None:
            return
        now = time.monotonic()
        # Keep completed cards briefly; no timer is required because subsequent
        # render/activity traffic naturally cleans them up.
        expired = [
            key
            for key, activity in self._activities.items()
            if activity.completed_at is not None and now - activity.completed_at > 3.0
        ]
        for key in expired:
            self._activities.pop(key, None)

        cards = []
        for activity in self._activities.values():
            rank_states = [activity.ranks.get(rank) for rank in range(context.size)]
            complete = all(state is not None and state.done for state in rank_states)
            progresses = [
                0.0 if state is None or state.progress is None else float(state.progress)
                for state in rank_states
            ]
            progress = sum(progresses) / float(max(1, context.size))
            progress_values = [
                "—" if state is None or state.progress is None else f"{100.0 * state.progress:.0f}%"
                for state in rank_states
            ]
            durations = [
                "—" if state is None or state.duration is None else f"{state.duration:.2f}s"
                for state in rank_states
            ]
            cards.append(
                {
                    "key": activity.key,
                    "view_id": activity.view_id,
                    "label": activity.label,
                    "determinate": activity.determinate,
                    "progress": 1.0 if complete else progress,
                    "progress_values": progress_values,
                    "duration_values": durations,
                    "elapsed": max(0.0, now - activity.first_started_at),
                    "complete": complete,
                    "details": self._detail_lines(activity),
                }
            )

        if self._publisher is not None:
            by_view: dict[str, list[dict[str, Any]]] = {}
            for card in cards:
                by_view.setdefault(str(card["view_id"]), []).append(card)
            for view_id, view_cards in by_view.items():
                self._publisher(
                    view_id,
                    {"type": "activities", "activities": view_cards},
                )
