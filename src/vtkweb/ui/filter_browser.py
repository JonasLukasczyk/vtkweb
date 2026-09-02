from trame.widgets import vuetify3 as v3


def build_filter_browser(
    state,
    ctrl,
    catalog,
) -> None:
    state.filter_browser_open = False

    with v3.VDialog(
        model_value=("filter_browser_open",),
        width=600,
    ):
        with v3.VCard(classes="pa-4"):
            v3.VCardTitle("Add Filter")
            v3.VCardText(
                "The filter browser is open."
            )
