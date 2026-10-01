from __future__ import annotations

import base64
from copy import deepcopy
from math import isfinite
from typing import Any

import numpy as np
from matplotlib import colormaps


DEFAULT_TRANSFER_FUNCTION = {
    "color": {
        "range": [0.0, 1.0],
        "control_points": [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
    },
    "opacity": {
        "range": [0.0, 1.0],
        "control_points": [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
    },
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


def color_points_from_colormap(name: str, samples: int = 16) -> list[list[float]]:
    """Sample a Matplotlib colormap into normalized color control points."""
    cmap = colormaps[name]
    points = []
    for t in np.linspace(0.0, 1.0, max(2, int(samples))):
        r, g, b, _ = cmap(float(t))
        points.append([float(t), float(r), float(g), float(b)])
    return points


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

        value = deepcopy(DEFAULT_TRANSFER_FUNCTION)
        if data_range is not None:
            mapping_range = [float(data_range[0]), float(data_range[1])]
            value["color"]["range"] = list(mapping_range)
            value["opacity"]["range"] = list(mapping_range)
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
        current["color"]["control_points"] = color_points_from_colormap(preset_name)
        self.set_data(array_name, current)

    def set_mapping_range(
        self,
        array_name: str,
        mapping_name: str,
        minimum: float,
        maximum: float,
    ) -> None:
        current = self.ensure(array_name)
        mapping = _mapping(current, mapping_name)
        mapping["range"] = _normalize_range([minimum, maximum])
        self.set_data(array_name, current)

    def rescale_mapping(self, array_name: str, mapping_name: str) -> None:
        data_range = self.rendering.get_global_array_range(array_name)
        if data_range is not None:
            self.set_mapping_range(array_name, mapping_name, *data_range)

    def set_color_control_point_component(
        self,
        array_name: str,
        point_index: int,
        component_index: int,
        value: float,
    ) -> None:
        current = self.ensure(array_name)
        points = [list(point) for point in current["color"]["control_points"]]
        point_index = int(point_index)
        component_index = int(component_index)
        if not (0 <= point_index < len(points)) or not (0 <= component_index <= 3):
            return
        points[point_index][component_index] = _clamp01(value)
        points.sort(key=lambda point: point[0])
        current["color"]["control_points"] = points
        self.set_data(array_name, current)

    def add_color_control_point(self, array_name: str) -> None:
        current = self.ensure(array_name)
        points = [list(point) for point in current["color"]["control_points"]]
        points.sort(key=lambda point: point[0])
        gap_index = max(
            range(len(points) - 1),
            key=lambda i: points[i + 1][0] - points[i][0],
        )
        left = points[gap_index]
        right = points[gap_index + 1]
        points.append([(a + b) * 0.5 for a, b in zip(left, right)])
        points.sort(key=lambda point: point[0])
        current["color"]["control_points"] = points
        self.set_data(array_name, current)

    def remove_color_control_point(self, array_name: str, point_index: int) -> None:
        current = self.ensure(array_name)
        points = [list(point) for point in current["color"]["control_points"]]
        if len(points) <= 2:
            return
        point_index = int(point_index)
        if 0 <= point_index < len(points):
            points.pop(point_index)
            current["color"]["control_points"] = points
            self.set_data(array_name, current)

    def set_opacity_control_point(
        self,
        array_name: str,
        point_index: int,
        x: float,
        opacity: float,
    ) -> None:
        current = self.ensure(array_name)
        points = [list(point) for point in current["opacity"]["control_points"]]
        point_index = int(point_index)
        if not (0 <= point_index < len(points)):
            return

        x = _clamp01(x)
        opacity = _clamp01(opacity)
        if point_index == 0:
            x = 0.0
        elif point_index == len(points) - 1:
            x = 1.0
        else:
            epsilon = 1.0e-6
            x = max(points[point_index - 1][0] + epsilon, x)
            x = min(points[point_index + 1][0] - epsilon, x)

        points[point_index] = [x, opacity]
        current["opacity"]["control_points"] = points
        self.set_data(array_name, current)

    def add_opacity_control_point(
        self,
        array_name: str,
        x: float,
        opacity: float,
    ) -> None:
        current = self.ensure(array_name)
        points = [list(point) for point in current["opacity"]["control_points"]]
        x = _clamp01(x)
        opacity = _clamp01(opacity)
        epsilon = 1.0e-6
        if x <= epsilon or x >= 1.0 - epsilon:
            return
        if any(abs(point[0] - x) <= epsilon for point in points):
            return
        points.append([x, opacity])
        points.sort(key=lambda point: point[0])
        current["opacity"]["control_points"] = points
        self.set_data(array_name, current)

    def remove_opacity_control_point(self, array_name: str, point_index: int) -> None:
        current = self.ensure(array_name)
        points = [list(point) for point in current["opacity"]["control_points"]]
        point_index = int(point_index)
        if point_index <= 0 or point_index >= len(points) - 1:
            return
        points.pop(point_index)
        current["opacity"]["control_points"] = points
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


def _mapping(value: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in {"color", "opacity"}:
        raise ValueError(f"Unknown transfer-function mapping: {name}")
    return value[name]


def _normalize_tf(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != {"color", "opacity"}:
        raise ValueError("transfer function must contain exactly color and opacity mappings")

    color = value["color"]
    opacity = value["opacity"]
    return {
        "color": {
            "range": _normalize_range(color.get("range")),
            "control_points": _normalize_points(
                color.get("control_points"), 4, "color"
            ),
        },
        "opacity": {
            "range": _normalize_range(opacity.get("range")),
            "control_points": _normalize_opacity_points(opacity.get("control_points")),
        },
    }


def _normalize_range(value) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("mapping range must be [minimum, maximum]")
    minimum = _finite_float(value[0])
    maximum = _finite_float(value[1])
    if maximum <= minimum:
        raise ValueError("mapping range maximum must be greater than minimum")
    return [minimum, maximum]


def _normalize_points(value, width: int, label: str) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{label} mapping requires at least two control points")
    points = []
    for raw_point in value:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != width:
            raise ValueError(f"{label} control points must have {width} components")
        points.append([_clamp01(component) for component in raw_point])
    points.sort(key=lambda point: point[0])
    return points


def _normalize_opacity_points(value) -> list[list[float]]:
    points = _normalize_points(value, 2, "opacity")
    points[0][0] = 0.0
    points[-1][0] = 1.0
    for index in range(1, len(points)):
        if points[index][0] <= points[index - 1][0]:
            raise ValueError("opacity control-point x coordinates must be strictly increasing")
    return points


def _finite_float(value: float) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError("transfer-function values must be finite")
    return result


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, _finite_float(value)))


def mapping_scalar(mapping: dict[str, Any], x: float) -> float:
    """Convert a normalized mapping coordinate to scalar space."""
    minimum, maximum = mapping["range"]
    return float(minimum) + float(x) * (float(maximum) - float(minimum))
