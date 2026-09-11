from __future__ import annotations

import base64
from copy import deepcopy
from math import isfinite
from typing import Any

import numpy as np
from matplotlib import colormaps


DEFAULT_TRANSFER_FUNCTION = {
    "control_points": [
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ],
}


def available_presets() -> list[dict[str, Any]]:
    """Return Matplotlib colormaps with small inline selector previews."""
    names = set(colormaps)
    visible = sorted(
        name for name in names if not (name.endswith("_r") and name[:-2] in names)
    )
    return [
        {
            "title": name,
            "value": name,
            "props": {"appendAvatar": _preset_preview_uri(name)},
        }
        for name in visible
    ]


def _preset_preview_uri(name: str, samples: int = 24) -> str:
    """Create a tiny SVG gradient preview without adding image files/assets."""
    cmap = colormaps[name]
    stops = []
    for index, t in enumerate(np.linspace(0.0, 1.0, max(2, int(samples)))):
        r, g, b, _ = cmap(float(t))
        color = f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"
        offset = 100.0 * index / (max(2, int(samples)) - 1)
        stops.append(f'<stop offset="{offset:.2f}%" stop-color="{color}"/>')
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="16" '
        'viewBox="0 0 96 16" preserveAspectRatio="none">'
        '<defs><linearGradient id="g">'
        + "".join(stops)
        + '</linearGradient></defs><rect width="96" height="16" fill="url(#g)"/></svg>'
    )
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def transfer_function_from_colormap(
    name: str,
    data_range: tuple[float, float] | list[float] = (0.0, 1.0),
    samples: int = 16,
) -> dict[str, Any]:
    """Sample a Matplotlib colormap directly into scalar-space TF points."""
    minimum, maximum = map(float, data_range)
    width = maximum - minimum
    cmap = colormaps[name]
    points = []
    for t in np.linspace(0.0, 1.0, max(2, int(samples))):
        r, g, b, _ = cmap(float(t))
        points.append(
            [
                minimum + float(t) * width,
                float(r),
                float(g),
                float(b),
                1.0,
            ]
        )
    return {"control_points": points}


class TransferFunctionManager:
    """Global transfer functions keyed only by array name."""

    def __init__(self, state, rendering) -> None:
        self.state = state
        self.rendering = rendering
        self.state.transfer_functions = {}
        self.state.tf_preset_items = available_presets()

    def clear(self) -> None:
        self.state.transfer_functions = {}

    def get(self, array_name: str) -> dict[str, Any] | None:
        value = self.state.transfer_functions.get(str(array_name))
        return deepcopy(value) if value is not None else None

    def ensure(
        self,
        array_name: str,
        data_range: tuple[float, float] | list[float] | None = None,
    ) -> dict[str, Any]:
        current = self.state.transfer_functions.get(array_name)
        if current is not None:
            return deepcopy(current)

        if data_range is None:
            value = deepcopy(DEFAULT_TRANSFER_FUNCTION)
        else:
            minimum, maximum = map(float, data_range)
            value = {
                "control_points": [
                    [minimum, 0.0, 0.0, 0.0, 1.0],
                    [maximum, 1.0, 1.0, 1.0, 1.0],
                ]
            }
        self.set_data(array_name, value)
        return self.get(array_name) or deepcopy(value)

    def set_data(self, array_name: str, value: dict[str, Any]) -> None:
        normalized = _normalize_tf(value)
        transfer_functions = dict(self.state.transfer_functions)
        transfer_functions[str(array_name)] = normalized
        self.state.transfer_functions = transfer_functions
        self._refresh(str(array_name))

    def apply_preset(self, array_name: str, preset_name: str) -> None:
        current = self.ensure(array_name)
        self.set_data(
            array_name,
            transfer_function_from_colormap(preset_name, _point_range(current)),
        )

    def set_range(self, array_name: str, minimum: float, maximum: float) -> None:
        current = self.ensure(array_name)
        points = [list(point) for point in current["control_points"]]
        old_minimum, old_maximum = _point_range(current)
        minimum = float(minimum)
        maximum = float(maximum)
        old_width = old_maximum - old_minimum
        new_width = maximum - minimum

        if abs(old_width) < 1.0e-20:
            count = len(points) - 1
            for index, point in enumerate(points):
                t = index / count if count else 0.0
                point[0] = minimum + t * new_width
        else:
            for point in points:
                t = (float(point[0]) - old_minimum) / old_width
                point[0] = minimum + t * new_width

        current["control_points"] = points
        self.set_data(array_name, current)

    def rescale(self, array_name: str) -> None:
        data_range = self.rendering.get_global_array_range(array_name)
        if data_range is not None:
            self.set_range(array_name, *data_range)

    def set_control_point_component(
        self,
        array_name: str,
        point_index: int,
        component_index: int,
        value: float,
    ) -> None:
        current = self.ensure(array_name)
        points = [list(point) for point in current["control_points"]]
        point_index = int(point_index)
        component_index = int(component_index)
        if not (0 <= point_index < len(points)):
            return
        if not (0 <= component_index <= 4):
            return
        points[point_index][component_index] = (
            _finite_float(value) if component_index == 0 else _clamp01(value)
        )
        points.sort(key=lambda point: point[0])
        current["control_points"] = points
        self.set_data(array_name, current)

    def add_control_point(self, array_name: str) -> None:
        current = self.ensure(array_name)
        points = [list(point) for point in current["control_points"]]
        points.sort(key=lambda point: point[0])

        gap_index = max(
            range(len(points) - 1),
            key=lambda i: points[i + 1][0] - points[i][0],
        )
        left = points[gap_index]
        right = points[gap_index + 1]
        midpoint = [(a + b) * 0.5 for a, b in zip(left, right)]
        midpoint[4] = 1.0 if left[4] == right[4] == 1.0 else midpoint[4]
        points.append(midpoint)
        points.sort(key=lambda point: point[0])
        current["control_points"] = points
        self.set_data(array_name, current)

    def remove_control_point(self, array_name: str, point_index: int) -> None:
        current = self.ensure(array_name)
        points = [list(point) for point in current["control_points"]]
        if len(points) <= 2:
            return
        point_index = int(point_index)
        if 0 <= point_index < len(points):
            points.pop(point_index)
            current["control_points"] = points
            self.set_data(array_name, current)

    def discover_node_outputs(self, node_id: str) -> None:
        node = self.rendering.pipeline.nodes[node_id]
        for output_port in range(node.processor.GetNumberOfOutputPorts()):
            arrays = self.rendering.get_arrays(node_id, output_port)
            for association in ("point", "cell"):
                for array_name in arrays[association]:
                    if array_name in self.state.transfer_functions:
                        continue
                    data_range = self.rendering.get_array_range(
                        node_id,
                        output_port,
                        array_name,
                        association,
                    )
                    if data_range is not None:
                        self.ensure(array_name, data_range)

    def _refresh(self, array_name: str) -> None:
        for representation in tuple(self.rendering.representations):
            color_by = representation.properties.get("color_by")
            if color_by and color_by[0] == array_name:
                self.rendering.refresh_representation(representation.id)


def _point_range(value: dict[str, Any]) -> tuple[float, float]:
    points = value["control_points"]
    return float(points[0][0]), float(points[-1][0])


def _normalize_tf(value: dict[str, Any]) -> dict[str, Any]:
    raw_points = value.get("control_points", [])
    if len(raw_points) < 2:
        raise ValueError("transfer function requires at least two control points")

    points = []
    for raw_point in raw_points:
        if len(raw_point) != 5:
            raise ValueError("control points must have format [value, r, g, b, o]")
        points.append(
            [
                _finite_float(raw_point[0]),
                _clamp01(raw_point[1]),
                _clamp01(raw_point[2]),
                _clamp01(raw_point[3]),
                _clamp01(raw_point[4]),
            ]
        )
    points.sort(key=lambda point: point[0])
    return {"control_points": points}


def _finite_float(value: float) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError("transfer-function scalar positions must be finite")
    return result


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
