from __future__ import annotations

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from vtkweb.catalog import AlgorithmCatalog


NODE_BROWSER_STYLE = """
.vtkweb-node-search {
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

.vtkweb-node-search:focus {
    border-color: #4f7df3;
}

.vtkweb-node-list {
    margin-top: 8px;

    max-height: 55vh;
    overflow-y: auto;
}

.vtkweb-node-item {
    padding: 6px 10px;

    border-radius: 4px;

    cursor: pointer;
}

.vtkweb-node-item:hover,
.vtkweb-node-item-selected {
    background: rgba(79, 125, 243, 0.18);
}

.vtkweb-node-title {
    font-size: 13px;
}

.vtkweb-node-class {
    margin-top: 1px;

    font-size: 11px;
    opacity: 0.5;
}
"""


def initialize_node_browser(
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

    state.node_catalog_items = items
    state.node_browser_items = items

    state.node_browser_open = False
    state.node_browser_query = ""
    state.node_browser_selected = 0

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    def open_node_browser() -> None:
        with state:
            state.node_browser_query = ""
            state.node_browser_items = state.node_catalog_items
            state.node_browser_selected = 0
            state.node_browser_open = True

    def close_node_browser() -> None:
        state.node_browser_open = False

    def set_node_browser_query(
        query: str,
    ) -> None:
        query = query or ""
        needle = query.casefold()

        items = [
            item
            for item in state.node_catalog_items
            if (
                not needle
                or needle in item["title"].casefold()
                or needle in item["class_name"].casefold()
            )
        ]

        with state:
            state.node_browser_query = query
            state.node_browser_items = items
            state.node_browser_selected = 0

    def set_node_browser_selected(
        index: int,
    ) -> None:
        state.node_browser_selected = int(index)

    def node_browser_keydown(
        key: str,
    ) -> None:
        items = state.node_browser_items

        if key == "ArrowDown" and items:
            state.node_browser_selected = min(
                state.node_browser_selected + 1,
                len(items) - 1,
            )

        elif key == "ArrowUp" and items:
            state.node_browser_selected = max(
                state.node_browser_selected - 1,
                0,
            )

        elif key == "Enter" and items:
            ctrl.insert_node(items[state.node_browser_selected]["value"])

        elif key == "Escape":
            close_node_browser()

    # -------------------------------------------------------------------------
    # Controller
    # -------------------------------------------------------------------------

    ctrl.open_node_browser = open_node_browser

    ctrl.close_node_browser = close_node_browser

    ctrl.set_node_browser_query = set_node_browser_query

    ctrl.set_node_browser_selected = set_node_browser_selected

    ctrl.node_browser_keydown = node_browser_keydown

    server.trigger("open_node_browser")(open_node_browser)


def build_node_browser(
    state,
    ctrl,
) -> None:
    with v3.VDialog(
        model_value=("node_browser_open",),
        width=600,
        update_modelValue="node_browser_open = $event",
    ):
        with v3.VCard(
            classes="pa-3",
            style="max-height:70vh;",
        ):
            html.Input(
                id="vtkweb-node-search",
                autofocus=True,
                placeholder="Add source or filter...",
                value=("node_browser_query",),
                classes="vtkweb-node-search",
                input=(
                    ctrl.set_node_browser_query,
                    "[$event.target.value]",
                ),
                keydown=(
                    ctrl.node_browser_keydown,
                    "[$event.key]",
                ),
            )

            with html.Div(
                classes="vtkweb-node-list",
            ):
                with html.Div(
                    v_for=("(item, index) in node_browser_items"),
                    key=("item.value",),
                    classes=(
                        (
                            "index === node_browser_selected "
                            "? "
                            "'vtkweb-node-item "
                            "vtkweb-node-item-selected' "
                            ": "
                            "'vtkweb-node-item'"
                        ),
                    ),
                    mouseenter=(
                        ctrl.set_node_browser_selected,
                        "[index]",
                    ),
                    click=(
                        ctrl.insert_node,
                        "[item.value]",
                    ),
                ):
                    html.Div(
                        "{{ item.title }}",
                        classes="vtkweb-node-title",
                    )

                    html.Div(
                        ("{{ item.category }} · {{ item.class_name }}"),
                        classes="vtkweb-node-class",
                    )
