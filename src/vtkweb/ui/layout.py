from __future__ import annotations

from trame.ui.vuetify3 import (
    SinglePageLayout,
)
from trame.widgets import client
from trame.widgets import vuetify3 as v3

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

        with layout.toolbar:
            build_title_bar_actions(ctrl)

        # ---------------------------------------------------------------------
        # Content
        # ---------------------------------------------------------------------

        with layout.content:
            client.Style(INSPECTOR_STYLE)

            client.Style(NODE_BROWSER_STYLE)

            client.Style(PIPELINE_VIEW_STYLE)

            # -----------------------------------------------------------------
            # Global keyboard shortcuts
            # -----------------------------------------------------------------

            client.ClientTriggers(
                mounted="""
                    window.__vtkwebGlobalKeydown = (event) => {
                        const target = event.target;
                        const tag =
                            target?.tagName?.toLowerCase();

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
                """,
                before_unmount="""
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

            with v3.VContainer(
                fluid=True,
                classes="fill-height pa-0",
            ):
                with v3.VRow(
                    no_gutters=True,
                    classes="fill-height",
                ):
                    build_pipeline_view(
                        state,
                        ctrl,
                    )

                    build_render_view(
                        state,
                        ctrl,
                        rendering,
                    )

                    build_inspector_view(
                        ctrl,
                    )
