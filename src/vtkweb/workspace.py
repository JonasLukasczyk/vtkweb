from __future__ import annotations

from uuid import uuid4


class WorkspaceManager:
    """Serializable recursive split layout for heterogeneous views."""

    def __init__(self, state) -> None:
        self.state = state
        self.state.workspace_root_id = None
        self.state.workspace_nodes = {}
        self.state.workspace_tiles = []
        self.state.workspace_splitters = []

    def create_workspace(self, *, container_id: str | None = None) -> str:
        container_id = container_id or uuid4().hex
        self.state.workspace_root_id = container_id
        self.state.workspace_nodes = {
            container_id: self._leaf(container_id),
        }
        self._sync_geometry()
        return container_id

    def clear(self) -> None:
        self.state.workspace_root_id = None
        self.state.workspace_nodes = {}
        self._sync_geometry()

    def split_container(
        self,
        container_id: str,
        orientation: str,
        *,
        ratio: float = 0.5,
        first_id: str | None = None,
        second_id: str | None = None,
    ) -> tuple[str, str]:
        orientation = orientation.lower()
        if orientation not in {"horizontal", "vertical"}:
            raise ValueError("orientation must be 'horizontal' or 'vertical'")

        ratio = self._clamp_ratio(ratio)
        nodes = dict(self.state.workspace_nodes)
        node = dict(nodes[container_id])
        if node["kind"] != "leaf":
            raise ValueError(f"Container is already split: {container_id}")

        first_id = first_id or uuid4().hex
        second_id = second_id or uuid4().hex
        if first_id in nodes or second_id in nodes or first_id == second_id:
            raise ValueError("Child container IDs must be unique")

        existing_view = node.get("view_id")
        nodes[first_id] = self._leaf(first_id, existing_view)
        nodes[second_id] = self._leaf(second_id)
        nodes[container_id] = {
            "id": container_id,
            "kind": "split",
            "orientation": orientation,
            "ratio": ratio,
            "first": first_id,
            "second": second_id,
        }
        self.state.workspace_nodes = nodes
        self._sync_geometry()
        return first_id, second_id

    def assign_view(self, container_id: str, view_id: str | None) -> None:
        nodes = dict(self.state.workspace_nodes)
        node = dict(nodes[container_id])
        if node["kind"] != "leaf":
            raise ValueError("Views can only be assigned to leaf containers")
        node["view_id"] = view_id
        nodes[container_id] = node
        self.state.workspace_nodes = nodes
        self._sync_geometry()

    def set_split_ratio(self, container_id: str, ratio: float) -> None:
        nodes = dict(self.state.workspace_nodes)
        node = dict(nodes[container_id])
        if node["kind"] != "split":
            raise ValueError(f"Container is not split: {container_id}")
        node["ratio"] = self._clamp_ratio(ratio)
        nodes[container_id] = node
        self.state.workspace_nodes = nodes
        self._sync_geometry()

    def container_for_view(self, view_id: str) -> str | None:
        for node_id, node in self.state.workspace_nodes.items():
            if node.get("kind") == "leaf" and node.get("view_id") == view_id:
                return node_id
        return None

    def unassign_view(self, view_id: str) -> None:
        container_id = self.container_for_view(view_id)
        if container_id is not None:
            self.assign_view(container_id, None)

    def _sync_geometry(self) -> None:
        root_id = self.state.workspace_root_id
        nodes = self.state.workspace_nodes
        tiles: list[dict] = []
        splitters: list[dict] = []

        if root_id is not None and root_id in nodes:
            self._walk_geometry(root_id, 0.0, 0.0, 100.0, 100.0, tiles, splitters)

        self.state.workspace_tiles = tiles
        self.state.workspace_splitters = splitters

    def _walk_geometry(
        self,
        node_id: str,
        x: float,
        y: float,
        width: float,
        height: float,
        tiles: list[dict],
        splitters: list[dict],
    ) -> None:
        node = self.state.workspace_nodes[node_id]
        if node["kind"] == "leaf":
            tiles.append(
                {
                    "container_id": node_id,
                    "view_id": node.get("view_id"),
                    "left": x,
                    "top": y,
                    "width": width,
                    "height": height,
                    "style": self._rect_style(x, y, width, height),
                }
            )
            return

        ratio = float(node["ratio"])
        orientation = node["orientation"]
        if orientation == "vertical":
            first_width = width * ratio
            second_width = width - first_width
            self._walk_geometry(node["first"], x, y, first_width, height, tiles, splitters)
            self._walk_geometry(node["second"], x + first_width, y, second_width, height, tiles, splitters)
            splitters.append(
                {
                    "id": node_id,
                    "orientation": orientation,
                    "left": x + first_width,
                    "top": y,
                    "width": width,
                    "height": height,
                    "parent_left": x,
                    "parent_top": y,
                    "parent_width": width,
                    "parent_height": height,
                    "style": f"left:{x + first_width}%;top:{y}%;height:{height}%;",
                }
            )
        else:
            first_height = height * ratio
            second_height = height - first_height
            self._walk_geometry(node["first"], x, y, width, first_height, tiles, splitters)
            self._walk_geometry(node["second"], x, y + first_height, width, second_height, tiles, splitters)
            splitters.append(
                {
                    "id": node_id,
                    "orientation": orientation,
                    "left": x,
                    "top": y + first_height,
                    "width": width,
                    "height": height,
                    "parent_left": x,
                    "parent_top": y,
                    "parent_width": width,
                    "parent_height": height,
                    "style": f"left:{x}%;top:{y + first_height}%;width:{width}%;",
                }
            )

    @staticmethod
    def _leaf(container_id: str, view_id: str | None = None) -> dict:
        return {
            "id": container_id,
            "kind": "leaf",
            "view_id": view_id,
        }

    @staticmethod
    def _rect_style(x: float, y: float, width: float, height: float) -> str:
        return f"left:{x}%;top:{y}%;width:{width}%;height:{height}%;"

    @staticmethod
    def _clamp_ratio(value: float) -> float:
        return max(0.1, min(0.9, float(value)))
