# Example vtkweb state showing heterogeneous tiled views.


def load(ctrl):
    ctrl.clear_state()

    vtk_view = ctrl.create_view(
        "vtk",
        name="VTK View",
        view_id="vtk_view",
    )
    dummy_view = ctrl.create_view(
        "dummy",
        name="Dummy View",
        view_id="dummy_view",
        message="Dummy backend for tile testing",
    )

    root = ctrl.create_workspace(container_id="root")
    left, right = ctrl.split_container(
        root,
        "vertical",
        ratio=0.6,
        first_id="left",
        second_id="right",
    )
    ctrl.assign_view_to_container(left, vtk_view)
    ctrl.assign_view_to_container(right, dummy_view)
    ctrl.set_active_view(vtk_view)

    ctrl.finish_state_load()
