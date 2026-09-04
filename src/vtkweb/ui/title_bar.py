from __future__ import annotations

from trame.widgets import vuetify3 as v3


def build_title_bar_actions(ctrl) -> None:
    """Build application actions shown on the right side of the title bar."""

    v3.VSpacer()

    v3.VBtn(
        "Save State",
        prepend_icon="mdi-content-save-outline",
        variant="text",
        click=ctrl.save_python_state,
    )

    v3.VBtn(
        "Load State",
        prepend_icon="mdi-folder-open-outline",
        variant="text",
        click=ctrl.load_python_state,
    )
