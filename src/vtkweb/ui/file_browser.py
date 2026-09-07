from __future__ import annotations

from pathlib import Path

from trame.widgets import html
from trame.widgets import vuetify3 as v3

from vtkweb.pipeline import PipelineGraph


FILE_BROWSER_STYLE = """
.vtkweb-file-browser-card {
    display: flex;
    flex-direction: column;
    height: min(72vh, 720px);
    min-height: 420px;
}

.vtkweb-file-browser-header {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 14px 8px;
}

.vtkweb-file-browser-title {
    flex: 0 0 auto;
    font-size: 15px;
    font-weight: 600;
}

.vtkweb-file-browser-path {
    flex: 1 1 auto;
    min-width: 0;
    padding: 6px 8px;
    border: 1px solid rgba(128,128,128,0.45);
    border-radius: 4px;
    outline: none;
    background: rgba(128,128,128,0.08);
    color: inherit;
    font: inherit;
    font-size: 12px;
}

.vtkweb-file-browser-path:focus {
    border-color: #4f7df3;
}

.vtkweb-file-browser-breadcrumbs {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 2px;
    min-height: 32px;
    padding: 0 12px 8px;
    overflow-x: auto;
    white-space: nowrap;
}

.vtkweb-file-browser-crumb {
    flex: 0 0 auto;
}

.vtkweb-file-browser-list {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    margin: 0 12px;
    border: 1px solid rgba(128,128,128,0.28);
    border-radius: 6px;
}

.vtkweb-file-browser-row {
    display: flex;
    align-items: center;
    gap: 8px;
    min-height: 34px;
    padding: 4px 10px;
    cursor: default;
    user-select: none;
}

.vtkweb-file-browser-row:hover {
    background: rgba(128,128,128,0.12);
}

.vtkweb-file-browser-row-selected {
    background: rgba(79,125,243,0.20);
}

.vtkweb-file-browser-icon {
    flex: 0 0 auto;
    opacity: 0.78;
}

.vtkweb-file-browser-name {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 12px;
}

.vtkweb-file-browser-kind {
    flex: 0 0 auto;
    font-size: 11px;
    opacity: 0.45;
}

.vtkweb-file-browser-empty,
.vtkweb-file-browser-error {
    padding: 14px;
    font-size: 12px;
    opacity: 0.65;
}

.vtkweb-file-browser-error {
    color: rgb(var(--v-theme-error));
    opacity: 1;
}

.vtkweb-file-browser-footer {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px 12px;
}

.vtkweb-file-browser-selection {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11px;
    opacity: 0.7;
}
"""


def _safe_resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _breadcrumbs(path: Path) -> list[dict[str, str]]:
    parts: list[dict[str, str]] = []

    # pathlib.parents is easiest to reason about by walking from anchor.
    current = Path(path.anchor) if path.anchor else Path("/")
    parts.append({"title": path.anchor or "/", "path": str(current)})

    relative_parts = path.parts[1:] if path.anchor else path.parts
    for part in relative_parts:
        current = current / part
        parts.append({"title": part, "path": str(current)})

    return parts


def initialize_file_browser(
    state,
    ctrl,
    pipeline: PipelineGraph,
) -> None:
    state.file_browser_open = False
    state.file_browser_node_id = None
    state.file_browser_property_name = None
    state.file_browser_property_label = ""
    state.file_browser_mode = "file"
    state.file_browser_current_dir = ""
    state.file_browser_entries = []
    state.file_browser_breadcrumbs = []
    state.file_browser_selected_path = ""
    state.file_browser_error = ""
    state.file_browser_purpose = "property"
    state.file_browser_save_name = "state.py"

    def load_directory(path: str) -> None:
        try:
            directory = _safe_resolve(path)

            if not directory.exists():
                raise FileNotFoundError(f"Directory does not exist: {directory}")
            if not directory.is_dir():
                raise NotADirectoryError(str(directory))

            entries: list[dict[str, object]] = []
            for child in directory.iterdir():
                try:
                    is_dir = child.is_dir()
                    is_file = child.is_file()
                except OSError:
                    continue

                if not (is_dir or is_file):
                    continue

                entries.append(
                    {
                        "name": child.name,
                        "path": str(child.resolve()),
                        "is_dir": is_dir,
                        "kind": "Folder" if is_dir else "File",
                    }
                )

            entries.sort(
                key=lambda item: (
                    not bool(item["is_dir"]),
                    str(item["name"]).casefold(),
                )
            )

            with state:
                state.file_browser_current_dir = str(directory)
                state.file_browser_entries = entries
                state.file_browser_breadcrumbs = _breadcrumbs(directory)
                state.file_browser_selected_path = ""
                state.file_browser_error = ""
        except (OSError, ValueError) as exc:
            state.file_browser_error = str(exc)

    def open_file_browser(
        node_id: str,
        name: str,
    ) -> None:
        if node_id not in pipeline.nodes:
            return

        property_state = pipeline.node_state(node_id).get("properties", {}).get(name)
        if not property_state or property_state.get("kind") != "str":
            return

        lower_name = name.casefold()
        mode = "file" if "file" in lower_name else "directory"
        current_value = str(property_state.get("value") or "").strip()

        start_dir: Path | None = None
        selected_path = ""

        if current_value:
            try:
                current_path = _safe_resolve(current_value)
                if current_path.is_dir():
                    start_dir = current_path
                    if mode == "directory":
                        selected_path = str(current_path)
                else:
                    parent = current_path.parent
                    if parent.exists() and parent.is_dir():
                        start_dir = parent
                    if current_path.exists() and current_path.is_file():
                        selected_path = str(current_path)
            except (OSError, ValueError):
                pass

        if start_dir is None:
            start_dir = Path.cwd().resolve()

        with state:
            state.file_browser_purpose = "property"
            state.file_browser_node_id = node_id
            state.file_browser_property_name = name
            state.file_browser_property_label = property_state.get("label", name)
            state.file_browser_mode = mode
            state.file_browser_open = True

        load_directory(str(start_dir))

        if selected_path:
            state.file_browser_selected_path = selected_path

    def open_state_file_browser() -> None:
        with state:
            state.file_browser_purpose = "state_open"
            state.file_browser_node_id = None
            state.file_browser_property_name = None
            state.file_browser_property_label = "Open state"
            state.file_browser_mode = "file"
            state.file_browser_open = True
        load_directory(state.file_browser_current_dir or str(Path.cwd().resolve()))

    def save_state_file_browser() -> None:
        with state:
            state.file_browser_purpose = "state_save"
            state.file_browser_node_id = None
            state.file_browser_property_name = None
            state.file_browser_property_label = "Save state"
            state.file_browser_mode = "file"
            state.file_browser_save_name = state.file_browser_save_name or "state.py"
            state.file_browser_open = True
        load_directory(state.file_browser_current_dir or str(Path.cwd().resolve()))

    def close_file_browser() -> None:
        state.file_browser_open = False

    def set_file_browser_open(value: bool) -> None:
        state.file_browser_open = bool(value)

    def browse_file_browser_directory(path: str) -> None:
        load_directory(path)

    def browse_file_browser_parent() -> None:
        current = state.file_browser_current_dir
        if not current:
            return

        directory = _safe_resolve(current)
        parent = directory.parent
        load_directory(str(parent))

    def select_file_browser_entry(
        path: str,
        is_dir: bool,
    ) -> None:
        # In file mode, folders are navigational rather than selectable.
        if bool(is_dir) and state.file_browser_mode == "file":
            state.file_browser_selected_path = ""
            return

        state.file_browser_selected_path = path
        if state.file_browser_purpose == "state_save" and not bool(is_dir):
            state.file_browser_save_name = Path(path).name

    def activate_file_browser_entry(
        path: str,
        is_dir: bool,
    ) -> None:
        if bool(is_dir):
            load_directory(path)
            if state.file_browser_mode == "directory":
                state.file_browser_selected_path = str(_safe_resolve(path))
            return

        if state.file_browser_mode == "file":
            state.file_browser_selected_path = path
            if state.file_browser_purpose == "state_save":
                state.file_browser_save_name = Path(path).name
            else:
                confirm_file_browser_selection()

    def confirm_file_browser_selection() -> None:
        purpose = state.file_browser_purpose

        if purpose == "state_save":
            filename = str(state.file_browser_save_name or "").strip()
            if not filename:
                state.file_browser_error = "Enter a file name."
                return
            try:
                selected_path = _safe_resolve(
                    Path(state.file_browser_current_dir) / filename
                )
                ctrl.save_python_state_file(str(selected_path))
            except (OSError, ValueError, Exception) as exc:
                state.file_browser_error = str(exc)
                return
            state.file_browser_open = False
            return

        selected = state.file_browser_selected_path
        if state.file_browser_mode == "directory" and not selected:
            selected = state.file_browser_current_dir
        if not selected:
            return

        try:
            selected_path = _safe_resolve(selected)
        except (OSError, ValueError) as exc:
            state.file_browser_error = str(exc)
            return

        if state.file_browser_mode == "file" and not selected_path.is_file():
            state.file_browser_error = "Please select a file."
            return
        if state.file_browser_mode == "directory" and not selected_path.is_dir():
            state.file_browser_error = "Please select a directory."
            return

        if purpose == "state_open":
            try:
                ctrl.open_python_state_file(str(selected_path))
            except Exception as exc:
                state.file_browser_error = str(exc)
                return
            state.file_browser_open = False
            return

        node_id = state.file_browser_node_id
        name = state.file_browser_property_name
        if not node_id or not name or node_id not in pipeline.nodes:
            return
        pipeline.set_property(node_id, name, str(selected_path))
        state.file_browser_open = False

    ctrl.open_file_browser = open_file_browser
    ctrl.open_python_state = open_state_file_browser
    ctrl.save_python_state = save_state_file_browser
    ctrl.close_file_browser = close_file_browser
    ctrl.set_file_browser_open = set_file_browser_open
    ctrl.browse_file_browser_directory = browse_file_browser_directory
    ctrl.browse_file_browser_parent = browse_file_browser_parent
    ctrl.select_file_browser_entry = select_file_browser_entry
    ctrl.activate_file_browser_entry = activate_file_browser_entry
    ctrl.confirm_file_browser_selection = confirm_file_browser_selection


def build_file_browser(
    state,
    ctrl,
) -> None:
    with v3.VDialog(
        model_value=("file_browser_open",),
        width=820,
        update_modelValue=(
            ctrl.set_file_browser_open,
            "[$event]",
        ),
    ):
        with v3.VCard(classes="vtkweb-file-browser-card"):
            with html.Div(classes="vtkweb-file-browser-header"):
                html.Div(
                    "{{ file_browser_purpose === 'state_save' ? 'Save state' : (file_browser_purpose === 'state_open' ? 'Open state' : (file_browser_mode === 'file' ? 'Select file' : 'Select folder')) }}",
                    classes="vtkweb-file-browser-title",
                )

                with v3.VBtn(
                    icon=True,
                    size="small",
                    variant="text",
                    title="Parent directory",
                    click=ctrl.browse_file_browser_parent,
                ):
                    v3.VIcon("mdi-arrow-up", size="small")

                html.Input(
                    classes="vtkweb-file-browser-path",
                    value=("file_browser_current_dir",),
                    change=(
                        ctrl.browse_file_browser_directory,
                        "[$event.target.value]",
                    ),
                    keydown_enter=(
                        ctrl.browse_file_browser_directory,
                        "[$event.target.value]",
                    ),
                )

            with html.Div(classes="vtkweb-file-browser-breadcrumbs"):
                with html.Template(
                    v_for=("(crumb, index) in file_browser_breadcrumbs",),
                    key=("crumb.path",),
                ):
                    v3.VIcon(
                        "mdi-chevron-right",
                        v_if="index > 0",
                        size="x-small",
                    )
                    v3.VBtn(
                        "{{ crumb.title }}",
                        size="x-small",
                        variant="text",
                        classes="vtkweb-file-browser-crumb",
                        click=(
                            ctrl.browse_file_browser_directory,
                            "[crumb.path]",
                        ),
                    )

            with html.Div(classes="vtkweb-file-browser-list"):
                html.Div(
                    "{{ file_browser_error }}",
                    v_if="file_browser_error",
                    classes="vtkweb-file-browser-error",
                )

                html.Div(
                    "This folder is empty.",
                    v_if="!file_browser_error && file_browser_entries.length === 0",
                    classes="vtkweb-file-browser-empty",
                )

                with html.Div(
                    v_for=("entry in file_browser_entries",),
                    key=("entry.path",),
                    classes=(
                        "file_browser_selected_path === entry.path "
                        "? 'vtkweb-file-browser-row vtkweb-file-browser-row-selected' "
                        ": 'vtkweb-file-browser-row'",
                    ),
                    click=(
                        ctrl.select_file_browser_entry,
                        "[entry.path, entry.is_dir]",
                    ),
                    dblclick=(
                        ctrl.activate_file_browser_entry,
                        "[entry.path, entry.is_dir]",
                    ),
                ):
                    v3.VIcon(
                        "{{ entry.is_dir ? 'mdi-folder' : 'mdi-file-outline' }}",
                        size="small",
                        classes="vtkweb-file-browser-icon",
                    )
                    html.Div(
                        "{{ entry.name }}",
                        classes="vtkweb-file-browser-name",
                    )
                    html.Div(
                        "{{ entry.kind }}",
                        classes="vtkweb-file-browser-kind",
                    )

            with html.Div(classes="vtkweb-file-browser-footer"):
                html.Input(
                    v_if="file_browser_purpose === 'state_save'",
                    classes="vtkweb-file-browser-path",
                    value=("file_browser_save_name",),
                    input="file_browser_save_name = $event.target.value",
                    keydown_enter=ctrl.confirm_file_browser_selection,
                    placeholder="state.py",
                )
                html.Div(
                    "{{ file_browser_purpose === 'state_save' "
                    "? file_browser_current_dir "
                    ": (file_browser_selected_path || "
                    "(file_browser_mode === 'directory' ? file_browser_current_dir : 'No file selected')) }}",
                    classes="vtkweb-file-browser-selection",
                )

                v3.VBtn(
                    "Cancel",
                    size="small",
                    variant="text",
                    click=ctrl.close_file_browser,
                )
                v3.VBtn(
                    "{{ file_browser_purpose === 'state_save' ? 'Save' : (file_browser_mode === 'file' ? 'Open' : 'Select') }}",
                    size="small",
                    variant="flat",
                    disabled=(
                        "file_browser_purpose !== 'state_save' && file_browser_mode === 'file' && !file_browser_selected_path",
                    ),
                    click=ctrl.confirm_file_browser_selection,
                )
