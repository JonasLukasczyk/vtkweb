from __future__ import annotations

from trame.ui.vuetify3 import (
    SinglePageWithDrawerLayout,
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
from vtkweb.ui.filter_browser import (
    FILTER_BROWSER_STYLE,
    build_filter_browser,
    initialize_filter_browser,
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
from vtkweb.ui.view_tab import (
    initialize_view_tab,
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

    initialize_view_tab(
        state,
        ctrl,
        pipeline,
        rendering,
    )

    initialize_filter_browser(
        server,
        catalog,
    )

    # -------------------------------------------------------------------------
    # Initial inspector state
    # -------------------------------------------------------------------------

    if (
        pipeline.active_node_id
        is not None
    ):
        ctrl.update_properties_state(
            pipeline.active_node_id
        )

        ctrl.update_representation_state(
            pipeline.active_node_id
        )

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------

    with SinglePageWithDrawerLayout(
        server,
        show_drawer=False,
        width=220,
    ) as layout:
        layout.title.set_text(
            "vtkweb"
        )

        # ---------------------------------------------------------------------
        # Drawer
        # ---------------------------------------------------------------------

        with layout.drawer:
            with v3.VList(
                density="compact",
                nav=True,
            ):
                v3.VListItem(
                    title=(
                        "Add Source / Filter"
                    ),
                    prepend_icon="mdi-plus",
                    click=(
                        ctrl.open_filter_browser
                    ),
                )

        # ---------------------------------------------------------------------
        # Content
        # ---------------------------------------------------------------------

        with layout.content:
            client.Style(
                INSPECTOR_STYLE
            )

            client.Style(
                FILTER_BROWSER_STYLE
            )

            client.Style(
                PIPELINE_VIEW_STYLE
            )

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
                                'open_filter_browser'
                            );

                            return;
                        }

                        if (
                            event.key === 'Delete' &&
                            !editing &&
                            !filter_browser_open
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
            # Filter browser
            # -----------------------------------------------------------------

            build_filter_browser(
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
                        pipeline,
                    )

                    build_render_view(
                        ctrl,
                        rendering,
                    )

                    build_inspector_view(
                        ctrl,
                    )
