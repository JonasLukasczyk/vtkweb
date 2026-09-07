from __future__ import annotations

from trame.widgets import vuetify3 as v3


def build_title_bar_actions(ctrl) -> None:
    """Build application actions shown on the right side of the title bar."""

    v3.VSpacer()

    v3.VBtn(
        icon="mdi-content-save-outline",
        variant="text",
        click=ctrl.save_python_state,
        density="compact",
    )

    v3.VBtn(
        icon="mdi-folder-open-outline",
        variant="text",
        click=ctrl.open_python_state,
        density="compact",
    )

    v3.VBtn(
        icon="mdi-play",
        variant="text",
        click=ctrl.execute_pipeline,
        density="compact",
        v_if="!pipeline_executing",
        title="Execute pipeline",
    )

    v3.VBtn(
        icon="mdi-pause",
        variant="text",
        click=ctrl.abort_pipeline,
        density="compact",
        v_else=True,
        title="Stop after current node",
    )
