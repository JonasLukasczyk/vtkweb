from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from vtkweb.catalog import AlgorithmCatalog


FILTER_BROWSER_STYLE = """
.vtkweb-filter-search {
    width: 100%;
    height: 36px;

    padding: 0 10px;

    border: 1px solid rgba(128, 128, 128, 0.5);
    border-radius: 4px;
    outline: none;

    background: rgba(128, 128, 128, 0.08);
    color: inherit;

    box-sizing: border-box;
}

.vtkweb-filter-search:focus {
    border-color: #4f7df3;
}

.vtkweb-filter-list {
    margin-top: 8px;

    max-height: 55vh;
    overflow-y: auto;
}

.vtkweb-filter-item {
    padding: 6px 10px;

    border-radius: 4px;

    cursor: pointer;
}

.vtkweb-filter-item:hover,
.vtkweb-filter-item-selected {
    background: rgba(79, 125, 243, 0.18);
}

.vtkweb-filter-title {
    font-size: 13px;
}

.vtkweb-filter-class {
    margin-top: 1px;

    font-size: 11px;
    opacity: 0.5;
}
"""


def initialize_filter_browser(
    server,
    catalog: AlgorithmCatalog,
) -> None:
    state = server.state
    ctrl = server.controller

    items = [
        {
            "title": item.label,
            "class_name": item.class_name,
            "category": item.category,
            "value": item.class_name,
        }
        for item in catalog.algorithms
    ]

    state.filter_catalog_items = items
    state.filter_browser_items = items

    state.filter_browser_open = False
    state.filter_browser_query = ""
    state.filter_browser_selected = 0

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    def open_filter_browser() -> None:
        with state:
            state.filter_browser_query = ""
            state.filter_browser_items = state.filter_catalog_items
            state.filter_browser_selected = 0
            state.filter_browser_open = True

    def close_filter_browser() -> None:
        state.filter_browser_open = False

    def set_filter_browser_query(
        query: str,
    ) -> None:
        query = query or ""
        needle = query.casefold()

        items = [
            item
            for item in state.filter_catalog_items
            if (
                not needle
                or needle in item["title"].casefold()
                or needle in item["class_name"].casefold()
            )
        ]

        with state:
            state.filter_browser_query = query
            state.filter_browser_items = items
            state.filter_browser_selected = 0

    def set_filter_browser_selected(
        index: int,
    ) -> None:
        state.filter_browser_selected = int(index)

    def filter_browser_keydown(
        key: str,
    ) -> None:
        items = state.filter_browser_items

        if key == "ArrowDown" and items:
            state.filter_browser_selected = min(
                state.filter_browser_selected + 1,
                len(items) - 1,
            )

        elif key == "ArrowUp" and items:
            state.filter_browser_selected = max(
                state.filter_browser_selected - 1,
                0,
            )

        elif key == "Enter" and items:
            ctrl.create_filter(items[state.filter_browser_selected]["value"])

        elif key == "Escape":
            close_filter_browser()

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    ctrl.open_filter_browser = open_filter_browser

    ctrl.close_filter_browser = close_filter_browser

    ctrl.set_filter_browser_query = set_filter_browser_query

    ctrl.set_filter_browser_selected = set_filter_browser_selected

    ctrl.filter_browser_keydown = filter_browser_keydown

    server.trigger("open_filter_browser")(open_filter_browser)


def build_filter_browser(
    state,
    ctrl,
) -> None:
    with v3.VDialog(
        model_value=("filter_browser_open",),
        width=600,
        update_modelValue="filter_browser_open = $event",
    ):
        with v3.VCard(
            classes="pa-3",
            style="max-height:70vh;",
        ):
            html.Input(
                id="vtkweb-filter-search",
                autofocus=True,
                placeholder="Add source or filter...",
                value=("filter_browser_query",),
                classes="vtkweb-filter-search",
                input=(
                    ctrl.set_filter_browser_query,
                    "[$event.target.value]",
                ),
                keydown=(
                    ctrl.filter_browser_keydown,
                    "[$event.key]",
                ),
            )

            with html.Div(
                classes="vtkweb-filter-list",
            ):
                with html.Div(
                    v_for=("(item, index) in filter_browser_items"),
                    key=("item.value",),
                    classes=(
                        (
                            "index === filter_browser_selected "
                            "? "
                            "'vtkweb-filter-item "
                            "vtkweb-filter-item-selected' "
                            ": "
                            "'vtkweb-filter-item'"
                        ),
                    ),
                    mouseenter=(
                        ctrl.set_filter_browser_selected,
                        "[index]",
                    ),
                    click=(
                        ctrl.create_filter,
                        "[item.value]",
                    ),
                ):
                    html.Div(
                        "{{ item.title }}",
                        classes="vtkweb-filter-title",
                    )

                    html.Div(
                        ("{{ item.category }} · {{ item.class_name }}"),
                        classes="vtkweb-filter-class",
                    )
