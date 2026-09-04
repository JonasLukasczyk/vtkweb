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
