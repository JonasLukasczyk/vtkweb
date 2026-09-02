from __future__ import annotations

from dataclasses import dataclass
import inspect

import vtk


@dataclass(frozen=True)
class AlgorithmDescriptor:
    class_name: str
    label: str
    category: str
    vtk_class: type[vtk.vtkAlgorithm]


class AlgorithmCatalog:
    def __init__(self) -> None:
        self.algorithms = self._discover()

    def _discover(self) -> list[AlgorithmDescriptor]:
        result = []

        for name in dir(vtk):
            vtk_class = getattr(vtk, name)

            if not inspect.isclass(vtk_class):
                continue

            try:
                if not issubclass(vtk_class, vtk.vtkAlgorithm):
                    continue
            except TypeError:
                continue

            if vtk_class is vtk.vtkAlgorithm:
                continue

            try:
                algorithm = vtk_class()
            except Exception:
                continue

            # For now, exclude sinks/writers with no outputs.
            if algorithm.GetNumberOfOutputPorts() == 0:
                continue

            category = (
                "Sources"
                if algorithm.GetNumberOfInputPorts() == 0
                else "Filters"
            )

            result.append(
                AlgorithmDescriptor(
                    class_name=name,
                    label=_make_label(name),
                    category=category,
                    vtk_class=vtk_class,
                )
            )

        return sorted(
            result,
            key=lambda item: (
                item.category,
                item.label,
            ),
        )

    def create(
        self,
        class_name: str,
    ) -> vtk.vtkAlgorithm:
        descriptor = next(
            item
            for item in self.algorithms
            if item.class_name == class_name
        )

        return descriptor.vtk_class()


def _make_label(class_name: str) -> str:
    name = class_name.removeprefix("vtk")

    result = []

    for i, char in enumerate(name):
        if (
            i
            and char.isupper()
            and not name[i - 1].isupper()
        ):
            result.append(" ")

        result.append(char)

    return "".join(result)
