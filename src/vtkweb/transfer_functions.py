from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from matplotlib import colormaps


DEFAULT_TRANSFER_FUNCTION = {
    "control_points": [
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ],
    "range": [0.0, 1.0],
}


def available_presets() -> list[dict[str, str]]:
    """Return Matplotlib colormaps suitable for the TF preset selector."""
    names = sorted(name for name in colormaps if not name.endswith("_r"))
    return [{"title": name, "value": name} for name in names]


def transfer_function_from_colormap(
    name: str,
    *,
    data_range: tuple[float, float] | list[float] = (0.0, 1.0),
    samples: int = 16,
) -> dict[str, Any]:
    """Sample a Matplotlib colormap into vtkweb's normalized TF format."""
    cmap = colormaps[name]
    ts = np.linspace(0.0, 1.0, max(2, int(samples)))
    points = []
    for t in ts:
        r, g, b, _ = cmap(float(t))
        points.append([float(t), float(r), float(g), float(b), 1.0])
    return {
        "control_points": points,
        "range": [float(data_range[0]), float(data_range[1])],
    }


class TransferFunctionManager:
    """Global transfer functions keyed only by array name."""

    def __init__(self, state, rendering=None) -> None:
        self.state = state
        self.rendering = rendering
        self.state.transfer_functions = {}
        self.state.tf_preset_items = available_presets()

    def clear(self) -> None:
        self.state.transfer_functions = {}

    def ensure(
        self,
        array_name: str,
        data_range: tuple[float, float] | list[float] | None = None,
    ) -> dict[str, Any]:
        current = self.state.transfer_functions.get(array_name)
        if current is not None:
            return deepcopy(current)
        value = deepcopy(DEFAULT_TRANSFER_FUNCTION)
        if data_range is not None:
            value["range"] = [float(data_range[0]), float(data_range[1])]
        self.set_data(array_name, value)
        return deepcopy(value)

    def set_data(self, array_name: str, value: dict[str, Any]) -> None:
        normalized = _normalize_tf(value)
        transfer_functions = dict(self.state.transfer_functions)
        transfer_functions[str(array_name)] = normalized
        self.state.transfer_functions = transfer_functions
        self._refresh(str(array_name))

    def delete(self, array_name: str) -> None:
        transfer_functions = dict(self.state.transfer_functions)
        transfer_functions.pop(array_name, None)
        self.state.transfer_functions = transfer_functions
        self._refresh(array_name)

    def apply_preset(self, array_name: str, preset_name: str) -> None:
        current = self.ensure(array_name)
        self.set_data(
            array_name,
            transfer_function_from_colormap(
                preset_name,
                data_range=current["range"],
            ),
        )

    def set_range(self, array_name: str, minimum: float, maximum: float) -> None:
        value = self.ensure(array_name)
        value["range"] = [float(minimum), float(maximum)]
        self.set_data(array_name, value)

    def set_control_point_component(
        self,
        array_name: str,
        point_index: int,
        component_index: int,
        value: float,
    ) -> None:
        tf = self.ensure(array_name)
        points = [list(point) for point in tf["control_points"]]
        point_index = int(point_index)
        component_index = int(component_index)
        if not 0 <= point_index < len(points):
            return
        if component_index == 4:
            return
        if not 0 <= component_index <= 3:
            return
        points[point_index][component_index] = _clamp01(value)
        points[point_index][4] = 1.0
        if component_index == 0:
            points.sort(key=lambda point: point[0])
        tf["control_points"] = points
        self.set_data(array_name, tf)

    def add_control_point(self, array_name: str) -> None:
        tf = self.ensure(array_name)
        points = [list(point) for point in tf["control_points"]]
        gap = max(
            range(len(points) - 1),
            key=lambda i: points[i + 1][0] - points[i][0],
        )
        left, right = points[gap], points[gap + 1]
        midpoint = [(a + b) * 0.5 for a, b in zip(left, right)]
        midpoint[4] = 1.0
        points.append(midpoint)
        points.sort(key=lambda point: point[0])
        tf["control_points"] = points
        self.set_data(array_name, tf)

    def remove_control_point(self, array_name: str, point_index: int) -> None:
        tf = self.ensure(array_name)
        points = [list(point) for point in tf["control_points"]]
        if len(points) <= 2:
            return
        point_index = int(point_index)
        if 0 <= point_index < len(points):
            points.pop(point_index)
            tf["control_points"] = points
            self.set_data(array_name, tf)

    def _refresh(self, array_name: str) -> None:
        if self.rendering is None:
            return
        for representation in tuple(self.rendering.representations):
            color_by = representation.properties.get("color_by")
            if color_by and color_by[0] == array_name:
                self.rendering.refresh_representation(representation.id)


def _normalize_tf(value: dict[str, Any]) -> dict[str, Any]:
    data_range = value.get("range", [0.0, 1.0])
    if len(data_range) != 2:
        raise ValueError("transfer-function range must contain two values")
    raw_points = value.get("control_points", [])
    if len(raw_points) < 2:
        raise ValueError("transfer function requires at least two control points")
    points = []
    for point in raw_points:
        if len(point) != 5:
            raise ValueError("control points must have format [t, r, g, b, o]")
        normalized = [_clamp01(component) for component in point[:4]] + [1.0]
        points.append(normalized)
    points.sort(key=lambda point: point[0])
    return {
        "control_points": points,
        "range": [float(data_range[0]), float(data_range[1])],
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
