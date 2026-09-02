from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import vtk


@dataclass
class PropertyDescriptor:
    name: str
    label: str
    kind: str
    value: Any
    size: int | None = None


IGNORED_PROPERTIES = {
    "AbortExecute",
    "Debug",
    "GlobalWarningDisplay",
    "ObjectName",
    "Progress",
    "ProgressObserver",
    "ReleaseDataFlag",
}


def inspect_properties(
    algorithm: vtk.vtkAlgorithm,
) -> list[PropertyDescriptor]:
    properties = []

    for getter_name in dir(algorithm):
        if not getter_name.startswith("Get"):
            continue

        name = getter_name[3:]

        if not name or name in IGNORED_PROPERTIES:
            continue

        getter = getattr(algorithm, getter_name, None)
        setter = getattr(algorithm, f"Set{name}", None)

        if not callable(getter) or not callable(setter):
            continue

        try:
            value = getter()
        except Exception:
            continue

        kind = _property_kind(
            algorithm,
            name,
            value,
        )

        if kind is None:
            continue

        properties.append(
            PropertyDescriptor(
                name=name,
                label=_make_label(name),
                kind=kind,
                value=value,
                size=len(value) if kind == "vector" else None,
            )
        )

    return sorted(
        properties,
        key=lambda prop: prop.label,
    )


def set_property(
    algorithm: vtk.vtkAlgorithm,
    descriptor: PropertyDescriptor,
    value,
) -> None:
    setter = getattr(
        algorithm,
        f"Set{descriptor.name}",
    )

    if descriptor.kind == "bool":
        setter(bool(value))

    elif descriptor.kind == "int":
        setter(int(value))

    elif descriptor.kind == "float":
        setter(float(value))

    elif descriptor.kind == "str":
        setter(str(value))

    elif descriptor.kind == "vector":
        setter(
            *[
                float(component)
                for component in value
            ]
        )


def _property_kind(
    algorithm,
    name: str,
    value,
) -> str | None:
    if (
        callable(getattr(algorithm, f"{name}On", None))
        and callable(getattr(algorithm, f"{name}Off", None))
    ):
        return "bool"

    if isinstance(value, bool):
        return "bool"

    if isinstance(value, int):
        return "int"

    if isinstance(value, float):
        return "float"

    if isinstance(value, str):
        return "str"

    if (
        isinstance(value, tuple)
        and value
        and all(
            isinstance(component, (int, float))
            for component in value
        )
    ):
        return "vector"

    return None


def _make_label(name: str) -> str:
    result = []

    for i, char in enumerate(name):
        if (
            i > 0
            and char.isupper()
            and not name[i - 1].isupper()
        ):
            result.append(" ")

        result.append(char)

    return "".join(result)
