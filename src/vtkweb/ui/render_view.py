from __future__ import annotations

from trame.widgets import client, html
from trame.widgets import vtk as vtk_widgets

from vtkweb.rendering import RenderManager, VTKRenderingBackend


WORKSPACE_STYLE = """
.vtkweb-workspace {
    position: relative;
    width: 100%;
    height: 100%;
    min-width: 0;
    min-height: 0;
    overflow: hidden;
    background: #111;
}

.vtkweb-workspace-tile,
.vtkweb-vtk-slot {
    position: absolute;
    box-sizing: border-box;
    overflow: hidden;
}

.vtkweb-workspace-tile {
    pointer-events: none;
    border: 1px solid rgba(128, 128, 128, 0.2);
    z-index: 20;
}

.vtkweb-tile-toolbar {
    position: absolute;
    top: 6px;
    right: 6px;
    display: flex;
    gap: 3px;
    z-index: 30;
    pointer-events: auto;
}

.vtkweb-tile-button {
    min-width: 24px;
    height: 24px;
    padding: 0 5px;
    border: 1px solid rgba(255, 255, 255, 0.25);
    border-radius: 3px;
    background: rgba(20, 20, 20, 0.75);
    color: #ddd;
    cursor: pointer;
    font: 11px sans-serif;
}

.vtkweb-dummy-view,
.vtkweb-empty-view {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 6px;
    font-family: sans-serif;
}

.vtkweb-dummy-view {
    color: #ddd;
    background: repeating-linear-gradient(
        135deg,
        #202020,
        #202020 10px,
        #252525 10px,
        #252525 20px
    );
}

.vtkweb-empty-view {
    color: #777;
    background: #181818;
}

.vtkweb-tile-splitter {
    position: absolute;
    z-index: 40;
    background: transparent;
}

.vtkweb-tile-splitter.vertical {
    width: 7px;
    margin-left: -3.5px;
    cursor: col-resize;
}

.vtkweb-tile-splitter.horizontal {
    height: 7px;
    margin-top: -3.5px;
    cursor: row-resize;
}

.vtkweb-tile-splitter::after {
    content: "";
    position: absolute;
    background: rgba(160, 160, 160, 0.35);
}

.vtkweb-tile-splitter.vertical::after {
    left: 3px;
    top: 0;
    bottom: 0;
    width: 1px;
}

.vtkweb-tile-splitter.horizontal::after {
    top: 3px;
    left: 0;
    right: 0;
    height: 1px;
}

.vtkweb-tile-splitter:hover::after {
    background: rgba(210, 210, 210, 0.8);
}
"""


def build_render_view(
    state,
    ctrl,
    rendering: RenderManager,
) -> None:
    """Build the generic tiled workspace and the fixed pool of VTK view widgets."""

    backend = rendering.backend
    if not isinstance(backend, VTKRenderingBackend):
        raise NotImplementedError(
            f"No render-view adapter for backend '{backend.name}'"
        )

    client.Style(WORKSPACE_STYLE)

    vtk_widgets_by_slot = {}

    def sync_slot_layout(**_):
        layout = {slot_id: None for slot_id in rendering.backend_slots}
        tiles_by_view = {
            tile.get("view_id"): tile
            for tile in state.workspace_tiles
            if tile.get("view_id") is not None
        }
        for view_id, value in state.views.items():
            if value.get("type") != "vtk":
                continue
            tile = tiles_by_view.get(view_id)
            if tile is None:
                continue
            slot_id = value["backend_id"]
            layout[slot_id] = {
                "view_id": view_id,
                "container_id": tile["container_id"],
                "style": tile["style"],
            }
        state.vtk_slot_layout = layout

    sync_slot_layout()

    @state.change("workspace_tiles", "views")
    def _sync_slots(**_):
        sync_slot_layout()

    def reset_render_view(view_id: str | None = None) -> None:
        if view_id is None or view_id not in state.views:
            return
        value = state.views[view_id]
        if value.get("type") != "vtk":
            return

        # Keyboard camera reset belongs to the focused VtkLocalView.  Reset
        # the client-side local view directly, matching the pre-tiling
        # behavior, rather than treating Space as a workspace-level action.
        widget = vtk_widgets_by_slot.get(value["backend_id"])
        if widget is not None:
            widget.reset_camera()
            widget.update()

    ctrl.trigger("render_view_reset")(reset_render_view)

    client.ClientTriggers(
        mounted="""
            window.__vtkwebStartTileResize = (splitter, event) => {
                event.preventDefault();
                event.stopPropagation();

                const pane = window.document.getElementById('vtkweb-right-pane');
                if (!pane) return;

                const rect = pane.getBoundingClientRect();
                const vertical = splitter.orientation === 'vertical';
                window.document.body.style.cursor = vertical ? 'col-resize' : 'row-resize';
                window.document.body.style.userSelect = 'none';

                const move = (moveEvent) => {
                    const position = vertical
                        ? ((moveEvent.clientX - rect.left) / rect.width) * 100.0
                        : ((moveEvent.clientY - rect.top) / rect.height) * 100.0;
                    const start = vertical ? splitter.parent_left : splitter.parent_top;
                    const extent = vertical ? splitter.parent_width : splitter.parent_height;
                    const ratio = Math.max(0.1, Math.min(0.9, (position - start) / extent));
                    trigger('set_split_ratio', [splitter.id, ratio]);
                };

                const up = () => {
                    window.removeEventListener('mousemove', move);
                    window.removeEventListener('mouseup', up);
                    window.document.body.style.cursor = '';
                    window.document.body.style.userSelect = '';
                };

                window.addEventListener('mousemove', move);
                window.addEventListener('mouseup', up);
            };
        """,
        before_unmount="""
            delete window.__vtkwebStartTileResize;
        """,
    )

    # Front-end trigger used by the splitter drag handler.
    ctrl.trigger("set_split_ratio")(
        lambda container_id, ratio: ctrl.set_split_ratio(container_id, ratio)
    )

    with html.Div(classes="vtkweb-workspace"):
        # The VTK widgets are created once, one per backend slot. Their logical
        # view assignment and geometry are driven entirely by serialized state.
        for slot_id in rendering.backend_slots:
            with html.Div(
                classes="vtkweb-vtk-slot",
                v_show=(f"vtk_slot_layout['{slot_id}'] !== null",),
                style=(
                    f"(vtk_slot_layout['{slot_id}']?.style || '') + 'z-index:10;'",
                ),
                click=(
                    ctrl.set_active_view,
                    f"[vtk_slot_layout['{slot_id}'].view_id]",
                ),
            ):
                widget = vtk_widgets.VtkLocalView(
                    backend.get_render_window(slot_id),
                    ref=f"render_view_{slot_id}",
                    tabindex=0,
                    style="height:100%;width:100%;outline:none;",
                    raw_attrs=[
                        f"@keydown.space.exact.prevent=\"trigger('render_view_reset', [vtk_slot_layout['{slot_id}'].view_id])\""
                    ],
                )
                vtk_widgets_by_slot[slot_id] = widget

        # Dummy and empty content are ordinary Vue/HTML and therefore need no
        # backend slots.
        with html.Div(
            v_for=("tile in workspace_tiles", "tile.container_id"),
            classes="vtkweb-workspace-tile",
            style=("tile.style",),
        ):
            with html.Div(
                v_if=("tile.view_id && views[tile.view_id]?.type === 'dummy'",),
                classes="vtkweb-dummy-view",
            ):
                html.Div("{{ views[tile.view_id]?.name || 'Dummy view' }}")
                html.Small("{{ views[tile.view_id]?.message || 'Dummy backend' }}")

            html.Div(
                "Empty tile",
                v_if=("!tile.view_id",),
                classes="vtkweb-empty-view",
            )

            with html.Div(classes="vtkweb-tile-toolbar"):
                html.Button(
                    "V",
                    title="Split vertically",
                    classes="vtkweb-tile-button",
                    click=(
                        ctrl.split_view_container,
                        "[tile.container_id, 'vertical']",
                    ),
                )
                html.Button(
                    "H",
                    title="Split horizontally",
                    classes="vtkweb-tile-button",
                    click=(
                        ctrl.split_view_container,
                        "[tile.container_id, 'horizontal']",
                    ),
                )

        html.Div(
            v_for=("splitter in workspace_splitters", "splitter.id"),
            classes=("['vtkweb-tile-splitter', splitter.orientation]",),
            style=("splitter.style",),
            raw_attrs=[
                '@mousedown="window.__vtkwebStartTileResize(splitter, $event)"'
            ],
        )

    @state.change("pipeline", "representations", "views", "workspace_tiles")
    def update_render_views(**_):
        for slot_id, widget in vtk_widgets_by_slot.items():
            if state.vtk_slot_layout.get(slot_id) is not None:
                widget.update()
