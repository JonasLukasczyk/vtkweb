from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable

import vtk


@dataclass
class PropertyDescriptor:
    name: str
    label: str
    kind: str
    value: Any
    size: int | None = None
    setter: Callable[[Any], None] | None = None


IGNORED_PROPERTIES = {
    "AbortExecute",
    "AbortOutput",
    "Debug",
    "GlobalWarningDisplay",
    "ObjectName",
    "Progress",
    "ProgressObserver",
    "ReferenceCount",
    "ReleaseDataFlag",
}


def inspect_properties(
    algorithm: vtk.vtkAlgorithm,
) -> list[PropertyDescriptor]:
    properties = []

    contour_values = _inspect_contour_values(algorithm)

    if contour_values is not None:
        properties.append(contour_values)

    for getter_name in dir(algorithm):
        if not getter_name.startswith("Get"):
            continue

        name = getter_name[3:]

        if not name or name in IGNORED_PROPERTIES:
            continue

        # Hidden because ContourValues represents this
        # information more naturally.
        if contour_values is not None and name in {
            "NumberOfContours",
            "ContourValues",
            "Values",
            "Value",
        }:
            continue

        getter = getattr(
            algorithm,
            getter_name,
            None,
        )

        setter = getattr(
            algorithm,
            f"Set{name}",
            None,
        )

        if not callable(getter) or not callable(setter):
            continue

        try:
            value = getter()
        except Exception:
            # Indexed getters such as GetValue(i)
            # land here and are handled separately.
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
                size=(len(value) if kind == "vector" else None),
            )
        )

    return sorted(
        properties,
        key=lambda prop: prop.label,
    )


def normalize_property_value(
    descriptor: PropertyDescriptor,
    value,
):
    """Normalize a user/model value to the descriptor's application type.

    This is intentionally independent of any value a VTK setter may later
    clamp or otherwise reinterpret. The returned value is the value stored in
    vtkweb's model.
    """

    if descriptor.kind == "bool":
        return bool(value)
    if descriptor.kind == "int":
        return int(value)
    if descriptor.kind == "float":
        return float(value)
    if descriptor.kind == "str":
        return None if value is None else str(value)
    if descriptor.kind == "scalar_list":
        return [float(item) for item in value]
    if descriptor.kind == "vector":
        converted = []
        for component, current in zip(value, descriptor.value):
            if isinstance(current, int) and not isinstance(current, bool):
                converted.append(int(component))
            else:
                converted.append(float(component))
        return tuple(converted) if isinstance(descriptor.value, tuple) else converted
    return value


def set_property(
    algorithm: vtk.vtkAlgorithm,
    descriptor: PropertyDescriptor,
    value,
) -> None:
    value = normalize_property_value(descriptor, value)

    if descriptor.setter is not None:
        descriptor.setter(value)
        return

    setter = getattr(algorithm, f"Set{descriptor.name}")
    if descriptor.kind == "vector":
        setter(*value)
    else:
        setter(value)


def _inspect_contour_values(
    algorithm: vtk.vtkAlgorithm,
) -> PropertyDescriptor | None:
    methods = (
        "GetNumberOfContours",
        "SetNumberOfContours",
        "GetValue",
        "SetValue",
    )

    if not all(callable(getattr(algorithm, name, None)) for name in methods):
        return None

    values = [
        float(algorithm.GetValue(i)) for i in range(algorithm.GetNumberOfContours())
    ]

    def set_values(new_values) -> None:
        values = [float(value) for value in new_values]

        algorithm.SetNumberOfContours(len(values))

        for i, value in enumerate(values):
            algorithm.SetValue(
                i,
                value,
            )

    return PropertyDescriptor(
        name="ContourValues",
        label="Contour Values",
        kind="scalar_list",
        value=values,
        size=len(values),
        setter=set_values,
    )


def _property_kind(
    algorithm,
    name: str,
    value,
) -> str | None:
    if callable(
        getattr(
            algorithm,
            f"{name}On",
            None,
        )
    ) and callable(
        getattr(
            algorithm,
            f"{name}Off",
            None,
        )
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

    # VTK string properties commonly return None until a value has been
    # assigned. FileName on vtkXMLImageDataReader is one example. Infer the
    # type from the wrapped setter signature so those properties still appear
    # before their first value has been set. Match an actual Python ``str``
    # annotation (rather than the substring "str", which can occur in VTK
    # object type names such as *Stream*) or the corresponding C++ char*.
    if value is None:
        setter = getattr(algorithm, f"Set{name}", None)
        doc = getattr(setter, "__doc__", "") or ""
        first_line = doc.splitlines()[0] if doc else ""

        python_string_arg = re.search(
            r"\([^\n]*:\s*str\s*(?:[,)]|$)",
            first_line,
        )
        cpp_string_arg = re.search(
            r"\bconst\s+char\s*\*",
            doc,
        )

        if python_string_arg or cpp_string_arg:
            return "str"

    if (
        isinstance(value, tuple)
        and value
        and all(isinstance(component, (int, float)) for component in value)
    ):
        return "vector"

    return None


def _make_label(name: str) -> str:
    result = []

    for i, char in enumerate(name):
        if i > 0 and char.isupper() and not name[i - 1].isupper():
            result.append(" ")

        result.append(char)

    return "".join(result)
