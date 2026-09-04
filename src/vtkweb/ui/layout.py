from __future__ import annotations

from trame.ui.vuetify3 import (
    SinglePageLayout,
)
from trame.widgets import client
from trame.widgets import html

from vtkweb.app_controller import (
    initialize_app_controller,
)
from vtkweb.catalog import (
    AlgorithmCatalog,
)
from vtkweb.pipeline import (
    PipelineGraph,
)
from vtkweb.rendering import (
    RenderManager,
)
from vtkweb.ui.node_browser import (
    NODE_BROWSER_STYLE,
    build_node_browser,
    initialize_node_browser,
)
from vtkweb.ui.inspector_style import (
    INSPECTOR_STYLE,
)
from vtkweb.ui.inspector_view import (
    build_inspector_view,
    initialize_inspector,
)
from vtkweb.ui.pipeline_view import (
    PIPELINE_VIEW_STYLE,
    build_pipeline_view,
)
from vtkweb.ui.properties_tab import (
    initialize_properties_tab,
)
from vtkweb.ui.render_view import (
    build_render_view,
)
from vtkweb.ui.representations_tab import (
    initialize_representations_tab,
)
from vtkweb.ui.title_bar import (
    build_title_bar_actions,
)


MAIN_LAYOUT_STYLE = """
.vtkweb-main-layout {
    width: 100%;
    height: 100%;

    display: flex;

    min-width: 0;
    min-height: 0;

    overflow: hidden;
}

.vtkweb-left-pane {
    width: 33%;
    height: 100%;

    min-width: 240px;
    min-height: 0;

    display: flex;
    flex-direction: column;

    overflow: hidden;
}

.vtkweb-right-pane {
    flex: 1 1 auto;

    height: 100%;

    min-width: 0;
    min-height: 0;

    overflow: hidden;
}

.vtkweb-main-splitter {
    width: 5px;
    flex: 0 0 5px;

    cursor: col-resize;

    position: relative;

    background: transparent;

    z-index: 10;
}

.vtkweb-main-splitter::after {
    content: "";

    position: absolute;

    top: 0;
    bottom: 0;

    left: 2px;

    width: 1px;

    background: rgba(
        128,
        128,
        128,
        0.3
    );
}

.vtkweb-main-splitter:hover::after {
    left: 1px;

    width: 3px;

    background: rgba(
        128,
        128,
        128,
        0.7
    );
}
"""


def build_ui(
    server,
    pipeline: PipelineGraph,
    rendering: RenderManager,
    catalog: AlgorithmCatalog,
) -> None:
    state = server.state
    ctrl = server.controller

    # -------------------------------------------------------------------------
    # Controllers / state
    # -------------------------------------------------------------------------

    initialize_app_controller(
        server,
        pipeline,
        rendering,
        catalog,
    )

    initialize_inspector(
        state,
        ctrl,
    )

    initialize_properties_tab(
        state,
        ctrl,
        pipeline,
    )

    initialize_representations_tab(
        state,
        ctrl,
        pipeline,
        rendering,
    )

    initialize_node_browser(
        server,
        catalog,
    )

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------

    with SinglePageLayout(
        server,
        theme=("theme", "dark"),
    ) as layout:
        layout.title.set_text("vtkweb")
        layout.icon.hide()

        layout.toolbar.density = "compact"

        with layout.toolbar:
            build_title_bar_actions(
                ctrl,
            )

        # ---------------------------------------------------------------------
        # Content
        # ---------------------------------------------------------------------

        with layout.content:
            client.Style(INSPECTOR_STYLE)

            client.Style(NODE_BROWSER_STYLE)

            client.Style(PIPELINE_VIEW_STYLE)

            client.Style(MAIN_LAYOUT_STYLE)

            # -----------------------------------------------------------------
            # Global client behavior
            # -----------------------------------------------------------------

            client.ClientTriggers(
                mounted="""
                    // ---------------------------------------------------------
                    // Global keyboard shortcuts
                    // ---------------------------------------------------------

                    window.__vtkwebGlobalKeydown = (
                        event
                    ) => {
                        const target =
                            event.target;

                        const tag =
                            target
                                ?.tagName
                                ?.toLowerCase();

                        const editing =
                            tag === 'input' ||
                            tag === 'textarea' ||
                            tag === 'select' ||
                            target?.isContentEditable;

                        if (
                            event.ctrlKey &&
                            event.code === 'Space'
                        ) {
                            event.preventDefault();

                            trigger(
                                'open_node_browser'
                            );

                            return;
                        }

                        if (
                            event.key === 'Delete' &&
                            !editing &&
                            !node_browser_open
                        ) {
                            event.preventDefault();

                            trigger(
                                'delete_active_node'
                            );
                        }
                    };

                    window.addEventListener(
                        'keydown',
                        window.__vtkwebGlobalKeydown
                    );

                    // ---------------------------------------------------------
                    // Main horizontal splitter
                    // ---------------------------------------------------------

                    const splitter =
                        window.document.getElementById(
                            'vtkweb-main-splitter'
                        );

                    const leftPane =
                        window.document.getElementById(
                            'vtkweb-left-pane'
                        );

                    if (
                        splitter &&
                        leftPane
                    ) {
                        window.__vtkwebSplitterDown = (
                            event
                        ) => {
                            event.preventDefault();

                            const parent =
                                splitter.parentElement;

                            const startX =
                                event.clientX;

                            const startWidth =
                                leftPane
                                    .getBoundingClientRect()
                                    .width;

                            const totalWidth =
                                parent
                                    .getBoundingClientRect()
                                    .width;

                            window.document.body.style.cursor =
                                'col-resize';

                            window.document.body.style.userSelect =
                                'none';

                            window.__vtkwebSplitterMove = (
                                moveEvent
                            ) => {
                                const delta =
                                    moveEvent.clientX -
                                    startX;

                                const width =
                                    startWidth +
                                    delta;

                                const minWidth =
                                    totalWidth *
                                    0.15;

                                const maxWidth =
                                    totalWidth *
                                    0.70;

                                const clampedWidth =
                                    Math.max(
                                        minWidth,
                                        Math.min(
                                            maxWidth,
                                            width
                                        )
                                    );

                                leftPane.style.width =
                                    `${clampedWidth}px`;
                            };

                            window.__vtkwebSplitterUp = () => {
                                window.removeEventListener(
                                    'mousemove',
                                    window.__vtkwebSplitterMove
                                );

                                window.removeEventListener(
                                    'mouseup',
                                    window.__vtkwebSplitterUp
                                );

                                window.document.body.style.cursor =
                                    '';

                                window.document.body.style.userSelect =
                                    '';
                            };

                            window.addEventListener(
                                'mousemove',
                                window.__vtkwebSplitterMove
                            );

                            window.addEventListener(
                                'mouseup',
                                window.__vtkwebSplitterUp
                            );
                        };

                        splitter.addEventListener(
                            'mousedown',
                            window.__vtkwebSplitterDown
                        );
                    }
                """,
                before_unmount="""
                    // ---------------------------------------------------------
                    // Keyboard cleanup
                    // ---------------------------------------------------------

                    if (
                        window.__vtkwebGlobalKeydown
                    ) {
                        window.removeEventListener(
                            'keydown',
                            window.__vtkwebGlobalKeydown
                        );

                        delete (
                            window.__vtkwebGlobalKeydown
                        );
                    }

                    // ---------------------------------------------------------
                    // Splitter cleanup
                    // ---------------------------------------------------------

                    const splitter =
                        window.document.getElementById(
                            'vtkweb-main-splitter'
                        );

                    if (
                        splitter &&
                        window.__vtkwebSplitterDown
                    ) {
                        splitter.removeEventListener(
                            'mousedown',
                            window.__vtkwebSplitterDown
                        );
                    }

                    if (
                        window.__vtkwebSplitterMove
                    ) {
                        window.removeEventListener(
                            'mousemove',
                            window.__vtkwebSplitterMove
                        );
                    }

                    if (
                        window.__vtkwebSplitterUp
                    ) {
                        window.removeEventListener(
                            'mouseup',
                            window.__vtkwebSplitterUp
                        );
                    }

                    delete window.__vtkwebSplitterDown;
                    delete window.__vtkwebSplitterMove;
                    delete window.__vtkwebSplitterUp;
                """,
            )

            # -----------------------------------------------------------------
            # Node browser
            # -----------------------------------------------------------------

            build_node_browser(
                state,
                ctrl,
            )

            # -----------------------------------------------------------------
            # Main layout
            # -----------------------------------------------------------------

            with html.Div(
                classes="vtkweb-main-layout",
            ):
                # -------------------------------------------------------------
                # Left pane
                # -------------------------------------------------------------

                with html.Div(
                    id="vtkweb-left-pane",
                    classes="vtkweb-left-pane",
                ):
                    # Pipeline
                    with html.Div(
                        style=("height:30%;min-height:0;overflow:hidden;"),
                    ):
                        build_pipeline_view(
                            state,
                            ctrl,
                        )

                    # Inspector
                    with html.Div(
                        style=("height:70%;min-height:0;overflow:hidden;"),
                    ):
                        build_inspector_view(
                            ctrl,
                        )

                # -------------------------------------------------------------
                # Main splitter
                # -------------------------------------------------------------

                html.Div(
                    id="vtkweb-main-splitter",
                    classes="vtkweb-main-splitter",
                )

                # -------------------------------------------------------------
                # Right pane
                # -------------------------------------------------------------

                with html.Div(
                    id="vtkweb-right-pane",
                    classes="vtkweb-right-pane",
                ):
                    build_render_view(
                        state,
                        ctrl,
                        rendering,
                    )
