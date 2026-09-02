from __future__ import annotations

from dataclasses import dataclass

import vtk


@dataclass
class InputArrayDescriptor:
    index: int
    label: str
    port: int
    connection: int
    value: str | None
    items: list[dict]


INPUT_ARRAY_COUNT_OVERRIDES: dict[str, int] = {}


def inspect_input_arrays(
    algorithm: vtk.vtkAlgorithm,
) -> list[InputArrayDescriptor]:
    if algorithm.GetNumberOfInputPorts() == 0:
        return []

    algorithm.UpdateInformation()

    count = (
        algorithm
        .GetNumberOfInputArraySpecifications()
    )

    # vtkContourFilter and similar filters expose
    # a single-array convenience API.
    if (
        count == 0
        and callable(
            getattr(
                algorithm,
                "SetInputArray",
                None,
            )
        )
    ):
        count = 1

    count = INPUT_ARRAY_COUNT_OVERRIDES.get(
        algorithm.GetClassName(),
        count,
    )

    descriptors = []

    for index in range(count):
        port = 0
        connection = 0
        association = None
        array_name = None

        if (
            index
            < algorithm
            .GetNumberOfInputArraySpecifications()
        ):
            info = (
                algorithm
                .GetInputArrayInformation(index)
            )

            if info is not None:
                if info.Has(
                    vtk.vtkAlgorithm.INPUT_PORT()
                ):
                    port = info.Get(
                        vtk.vtkAlgorithm.INPUT_PORT()
                    )

                if info.Has(
                    vtk.vtkAlgorithm.INPUT_CONNECTION()
                ):
                    connection = info.Get(
                        vtk.vtkAlgorithm.INPUT_CONNECTION()
                    )

                if info.Has(
                    vtk.vtkDataObject.FIELD_ASSOCIATION()
                ):
                    association = info.Get(
                        vtk.vtkDataObject.FIELD_ASSOCIATION()
                    )

                if info.Has(
                    vtk.vtkDataObject.FIELD_NAME()
                ):
                    array_name = info.Get(
                        vtk.vtkDataObject.FIELD_NAME()
                    )

        descriptors.append(
            InputArrayDescriptor(
                index=index,
                label=(
                    "Input Array"
                    if count == 1
                    else f"Input Array {index}"
                ),
                port=port,
                connection=connection,
                value=_encode_value(
                    association,
                    array_name,
                ),
                items=_available_arrays(
                    algorithm,
                    port,
                    connection,
                ),
            )
        )

    return descriptors


def set_input_array(
    algorithm: vtk.vtkAlgorithm,
    descriptor: InputArrayDescriptor,
    value: str,
) -> None:
    association_name, array_name = (
        value.split(":", 1)
    )

    # Prefer the filter's convenience API when
    # available. vtkContourFilter takes this path.
    if (
        descriptor.index == 0
        and association_name == "point"
        and callable(
            getattr(
                algorithm,
                "SetInputArray",
                None,
            )
        )
    ):
        algorithm.SetInputArray(
            array_name
        )
        return

    association = (
        vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
        if association_name == "point"
        else
        vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
    )

    algorithm.SetInputArrayToProcess(
        descriptor.index,
        descriptor.port,
        descriptor.connection,
        association,
        array_name,
    )


def _available_arrays(
    algorithm: vtk.vtkAlgorithm,
    port: int,
    connection: int,
) -> list[dict]:
    upstream = algorithm.GetInputAlgorithm(
        port,
        connection,
    )

    if upstream is not None:
        upstream.Update()

    data = algorithm.GetInputDataObject(
        port,
        connection,
    )

    if data is None:
        return []

    items = []

    point_data = data.GetPointData()

    for i in range(
        point_data.GetNumberOfArrays()
    ):
        name = point_data.GetArrayName(i)

        if name:
            items.append(
                {
                    "title": f"{name} (Point)",
                    "value": f"point:{name}",
                }
            )

    cell_data = data.GetCellData()

    for i in range(
        cell_data.GetNumberOfArrays()
    ):
        name = cell_data.GetArrayName(i)

        if name:
            items.append(
                {
                    "title": f"{name} (Cell)",
                    "value": f"cell:{name}",
                }
            )

    return items


def _encode_value(
    association: int | None,
    array_name: str | None,
) -> str | None:
    if array_name is None:
        return None

    if (
        association
        == vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS
    ):
        return f"point:{array_name}"

    if (
        association
        == vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS
    ):
        return f"cell:{array_name}"

    return None
