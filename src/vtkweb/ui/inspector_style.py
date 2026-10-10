INSPECTOR_STYLE = """
/* Icon-only inspector tabs: share available width without Vuetify's
   default text-tab minimum width. */
.vtkweb-inspector-icon-tabs .v-slide-group__content {
    width: 100%;
}
.vtkweb-inspector-icon-tabs .v-tab {
    min-width: 0 !important;
    flex: 1 1 0;
    padding-inline: 4px;
}

.vtkweb-section-title {
    margin-bottom: 6px;
    font-size: 12px;
    font-weight: 600;
    opacity: 0.8;
}

.vtkweb-prop-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 100%;
    min-width: 0;
}

.vtkweb-prop-item {
    width: 100%;
    min-width: 0;
}

.vtkweb-input-box,
.vtkweb-vector-box,
.vtkweb-select-box,
.vtkweb-list-row,
.vtkweb-bool-row,
.vtkweb-color-box {
    width: 100%;
    min-width: 0;
    min-height: 28px;
    border: 1px solid rgba(128,128,128,0.5);
    border-radius: 4px;
    background: rgba(128,128,128,0.08);
    box-sizing: border-box;
}

.vtkweb-input-box:hover,
.vtkweb-vector-box:hover,
.vtkweb-select-box:hover,
.vtkweb-color-box:hover {
    border-color: rgba(128,128,128,0.8);
}

.vtkweb-input-box:focus-within,
.vtkweb-vector-box:focus-within,
.vtkweb-select-box:focus-within,
.vtkweb-list-row:focus-within,
.vtkweb-color-box:focus-within {
    border-color: #4f7df3;
    background: rgba(79,125,243,0.06);
}

.vtkweb-control-label {
    flex: 0 0 auto;

    padding: 0 6px 0 8px;

    font-family: inherit;
    font-size: 12px;
    font-weight: 400;
    line-height: 26px;

    opacity: 0.82;

    white-space: nowrap;
    user-select: none;
}

.vtkweb-input-box {
    display: flex;
    align-items: center;
    height: 28px;
    overflow: hidden;
}

.vtkweb-input-box input {
    flex: 1 1 auto;
    min-width: 0;
    height: 26px;
    padding: 0 8px;
    border: 0;
    outline: 0;
    background: transparent;
    color: inherit;
    font: inherit;
    font-size: 12px;
    text-align: right;
    appearance: textfield;
    -moz-appearance: textfield;
}

.vtkweb-string-box {
    display: flex;
    align-items: flex-start;
    width: 100%;
    min-width: 0;
    min-height: 28px;
    border: 1px solid rgba(128,128,128,0.5);
    border-radius: 4px;
    background: rgba(128,128,128,0.08);
    box-sizing: border-box;
    overflow: hidden;
}

.vtkweb-string-box:hover {
    border-color: rgba(128,128,128,0.8);
}

.vtkweb-string-box:focus-within {
    border-color: #4f7df3;
    background: rgba(79,125,243,0.06);
}

.vtkweb-string-label {
    padding-top: 1px;
}

.vtkweb-string-input {
    flex: 1 1 auto;
    min-width: 0;
}

.vtkweb-string-input .v-input__control,
.vtkweb-string-input .v-field,
.vtkweb-string-input .v-field__field {
    min-height: 26px;
}

.vtkweb-string-input .v-field {
    padding: 0;
    background: transparent;
}

.vtkweb-string-input .v-field__input {
    min-height: 26px;
    padding: 4px 6px;
    font-size: 12px;
    line-height: 18px;
    text-align: left;
}

.vtkweb-string-input .v-field__outline,
.vtkweb-string-input .v-field__overlay {
    display: none;
}

.vtkweb-file-picker-button {
    flex: 0 0 auto;
    align-self: flex-start;
    min-width: 28px !important;
    width: 28px;
    height: 28px !important;
    margin: 0;
    border-radius: 0;
}

.vtkweb-vector-box {
    display: flex;
    align-items: center;
    height: 28px;
    overflow: hidden;
}

.vtkweb-vector-fields {
    flex: 1 1 auto;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    min-width: 0;
    height: 100%;
}

.vtkweb-vector-fields input {
    min-width: 0;
    width: 100%;
    height: 100%;
    padding: 0 5px;
    border: 0;
    border-left: 1px solid rgba(128,128,128,0.25);
    outline: 0;
    background: transparent;
    color: inherit;
    font: inherit;
    font-size: 12px;
    text-align: right;
    box-sizing: border-box;
    appearance: textfield;
    -moz-appearance: textfield;
}

.vtkweb-bool-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 28px;
    padding: 0 8px 0 0;
}

.vtkweb-select-box {
    display: flex;
    align-items: center;
    height: 28px;
    overflow: hidden;
}

.vtkweb-compact-select {
    flex: 1 1 auto;
    min-width: 0;
    width: 0;
}

.vtkweb-compact-select .v-field {
    min-height: 26px;
    height: 26px;
    padding: 0;
    background: transparent;
}

.vtkweb-compact-select .v-field__input {
    min-height: 26px;
    height: 26px;
    padding: 0 2px 0 6px;
    font-size: 12px;
}

.vtkweb-compact-select .v-field__append-inner {
    align-self: stretch;
    min-height: 26px;
    height: 26px;
    align-items: center;
    padding-top: 0;
}

.vtkweb-compact-select .v-field__append-inner > .v-icon {
    align-self: center;
    margin-top: 0;
}

.vtkweb-compact-select .v-input__control {
    min-height: 26px;
}

.vtkweb-compact-select .v-field__outline,
.vtkweb-compact-select .v-field__overlay {
    display: none;
}


/* VSelect menus are teleported outside the inspector DOM, so this selector
 * intentionally targets select menus globally. Keep their rows as compact as
 * the 28px inspector controls. */
.v-select__content .v-list {
    padding-top: 2px;
    padding-bottom: 2px;
}

.v-select__content .v-list-item {
    min-height: 30px;
    padding-top: 0;
    padding-bottom: 0;
}

.v-select__content .v-list-item-title {
    font-size: 12px;
    line-height: 20px;
}

.vtkweb-list-property {
    display: flex;
    flex-direction: column;
    gap: 4px;
    width: 100%;
}

.vtkweb-list-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 26px;
    padding-left: 8px;
    font-size: 12px;
    opacity: 0.82;
}

.vtkweb-list-add {
    height: 24px;
    padding: 0 8px;
    border: 1px solid rgba(128,128,128,0.5);
    border-radius: 4px;
    background: rgba(128,128,128,0.08);
    color: inherit;
    cursor: pointer;
}

.vtkweb-list-row {
    display: flex;
    align-items: center;
    height: 28px;
    overflow: hidden;
}

.vtkweb-list-index {
    padding: 0 7px;
    font-size: 11px;
    opacity: 0.45;
    user-select: none;
}

.vtkweb-list-inline {
    display: flex;
    align-items: center;

    width: 100%;
    min-width: 0;
    height: 28px;

    border: 1px solid rgba(128,128,128,0.5);
    border-radius: 4px;

    background: rgba(128,128,128,0.08);

    box-sizing: border-box;
    overflow: hidden;
}

.vtkweb-list-inline:hover {
    border-color: rgba(128,128,128,0.8);
}

.vtkweb-list-inline:focus-within {
    border-color: #4f7df3;
    background: rgba(79,125,243,0.06);
}

.vtkweb-list-inline-values {
    flex: 1 1 auto;

    display: grid;
    grid-auto-flow: column;
    grid-auto-columns: minmax(44px, 1fr);

    min-width: 0;
    height: 100%;

    overflow-x: auto;
}

.vtkweb-list-inline-values input {
    min-width: 44px;
    width: 100%;
    height: 100%;

    padding: 0 5px;

    border: 0;
    border-left: 1px solid rgba(128,128,128,0.25);
    outline: 0;

    background: transparent;
    color: inherit;

    font: inherit;
    font-size: 12px;
    font-weight: 400;
    text-align: right;

    box-sizing: border-box;

    appearance: textfield;
    -moz-appearance: textfield;
}

.vtkweb-list-inline-values input::-webkit-inner-spin-button,
.vtkweb-list-inline-values input::-webkit-outer-spin-button {
    -webkit-appearance: none;
    margin: 0;
}

.vtkweb-list-inline-button {
    flex: 0 0 28px;

    width: 28px;
    height: 100%;

    padding: 0;

    border: 0;
    border-left: 1px solid rgba(128,128,128,0.25);

    background: transparent;
    color: inherit;

    font: inherit;
    font-size: 14px;
    font-weight: 400;

    cursor: pointer;
}

.vtkweb-list-inline-button:hover {
    background: rgba(128,128,128,0.12);
}

.vtkweb-list-inline-button:disabled {
    opacity: 0.3;
    cursor: default;
}

.vtkweb-list-row input {
    flex: 1 1 auto;
    min-width: 0;
    height: 26px;
    padding: 0 8px;
    border: 0;
    outline: 0;
    background: transparent;
    color: inherit;
    font: inherit;
    font-size: 12px;
    text-align: right;
}

.vtkweb-list-remove {
    width: 28px;
    align-self: stretch;
    border: 0;
    border-left: 1px solid rgba(128,128,128,0.25);
    background: transparent;
    color: inherit;
    cursor: pointer;
}

.vtkweb-range-row {
    display: grid;
    grid-template-columns:
        minmax(0,1fr)
        minmax(0,1fr)
        auto;
    gap: 6px;
    width: 100%;
    margin-top: 6px;
}

.vtkweb-range-input {
    width: 100%;
    min-width: 0;
    height: 28px;
    padding: 0 7px;
    border: 1px solid rgba(128,128,128,0.5);
    border-radius: 4px;
    outline: none;
    background: rgba(128,128,128,0.08);
    color: inherit;
    font-size: 12px;
    text-align: right;
    box-sizing: border-box;
}

.vtkweb-color-box {
    display: flex;
    align-items: center;
    height: 32px;
    padding-left: 8px;
    overflow: hidden;
}

.vtkweb-color-box span {
    flex: 1 1 auto;
    font-size: 12px;
    opacity: 0.82;
}

.vtkweb-color-box input {
    width: 64px;
    height: 30px;
    padding: 0;
    border: 0;
    outline: 0;
    background: transparent;
    cursor: pointer;
}


.vtkweb-property-group {
    width: 100%;
    margin-top: 6px;
    border: 1px solid rgba(128,128,128,0.28);
    border-radius: 5px;
    background: rgba(128,128,128,0.025);
    box-sizing: border-box;
    overflow: hidden;
}

.vtkweb-property-group:first-child {
    margin-top: 0;
}

.vtkweb-property-group-header {
    display: flex;
    align-items: center;
    gap: 6px;
    min-height: 30px;
    padding: 0 8px;
    cursor: pointer;
    user-select: none;
    font-size: 12px;
    font-weight: 600;
    opacity: 0.86;
    list-style: none;
}

.vtkweb-property-group-header::-webkit-details-marker {
    display: none;
}

.vtkweb-property-group-header::after {
    content: "›";
    margin-left: auto;
    font-size: 16px;
    line-height: 1;
    opacity: 0.55;
    transform: rotate(0deg);
    transition: transform 120ms ease;
}

.vtkweb-property-group[open] > .vtkweb-property-group-header::after {
    transform: rotate(90deg);
}

.vtkweb-property-group-header:hover {
    background: rgba(128,128,128,0.08);
}

.vtkweb-property-group-icon {
    opacity: 0.68;
}

.vtkweb-property-group-body {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 5px 6px 6px;
    border-top: 1px solid rgba(128,128,128,0.18);
}

.vtkweb-view-property {
    display: contents;
}

.vtkweb-representation-cards {
    display: flex;
    flex-direction: column;
    gap: 8px;

    width: 100%;
}

.vtkweb-representation-card {
    display: flex;
    flex-direction: column;

    width: 100%;

    padding: 8px;

    border: 1px solid rgba(128,128,128,0.35);
    border-radius: 5px;

    background: rgba(128,128,128,0.05);

    box-sizing: border-box;
}

.vtkweb-representation-header {
    display: flex;
    align-items: center;

    height: 24px;
    margin-bottom: 6px;
}

.vtkweb-representation-title {
    flex: 1 1 auto;

    font-size: 12px;
    font-weight: 600;
    opacity: 0.85;
}

.vtkweb-representation-remove {
    width: 24px;
    height: 24px;

    padding: 0;

    border: 0;
    border-radius: 3px;

    background: transparent;
    color: inherit;

    font-size: 16px;
    line-height: 24px;

    cursor: pointer;
    opacity: 0.55;
}

.vtkweb-representation-remove:hover {
    background: rgba(255,255,255,0.08);
    opacity: 1;
}


.vtkweb-tf-section-title {
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px solid rgba(128,128,128,0.25);
    font-size: 12px;
    font-weight: 600;
}

.vtkweb-tf-table {
    width: 100%;
    margin-top: 8px;
    border-collapse: collapse;
    font-size: 12px;
}

.vtkweb-tf-table th,
.vtkweb-tf-table td {
    padding: 2px;
    text-align: left;
}

.vtkweb-tf-help {
    margin: 6px 0;
    font-size: 11px;
    line-height: 1.35;
    opacity: 0.7;
}

.vtkweb-opacity-editor {
    width: 100%;
    height: 150px;
    display: block;
    border: 1px solid rgba(128,128,128,0.45);
    border-radius: 4px;
    box-sizing: border-box;
    touch-action: none;
    cursor: crosshair;
}

.vtkweb-opacity-bg {
    fill: rgba(128,128,128,0.06);
    pointer-events: none;
}

.vtkweb-opacity-grid {
    stroke: rgba(128,128,128,0.2);
    stroke-width: 1;
    vector-effect: non-scaling-stroke;
    pointer-events: none;
}

.vtkweb-opacity-line {
    fill: none;
    stroke: currentColor;
    stroke-width: 2;
    vector-effect: non-scaling-stroke;
    pointer-events: none;
}

.vtkweb-opacity-point {
    fill: currentColor;
    stroke: rgba(255,255,255,0.75);
    stroke-width: 1.5;
    vector-effect: non-scaling-stroke;
    cursor: grab;
}

.vtkweb-opacity-point:active {
    cursor: grabbing;
}


/* The generic select-box clips overflow, hiding the custom preset menu. */
.vtkweb-select-box.vtkweb-colormap-select-box {
    overflow: visible;
    position: relative;
    z-index: 2;
}
.vtkweb-select-box.vtkweb-colormap-select-box:has(.vtkweb-colormap-dropdown[open]) {
    z-index: 100;
}

/* Custom native/Vue colormap selector. Unlike VSelect, the gradient is
 * painted on our own button elements, not forwarded through Vuetify props. */
.vtkweb-colormap-dropdown { position: relative; min-width: 0; flex: 1 1 auto; }
.vtkweb-colormap-dropdown-trigger {
    display: flex; align-items: center; justify-content: space-between;
    gap: 8px; min-height: 32px; padding: 3px 8px;
    border-radius: 4px; cursor: pointer; list-style: none;
    color: rgba(var(--v-theme-on-surface), .9);
    border-bottom: 1px solid rgba(var(--v-theme-on-surface), .38);
    font-size: 14px;
}
.vtkweb-colormap-dropdown-trigger::-webkit-details-marker { display: none; }
.vtkweb-colormap-dropdown-trigger:hover { background: rgba(var(--v-theme-on-surface), .05); }
.vtkweb-colormap-dropdown[open] .vtkweb-colormap-dropdown-trigger { border-bottom-color: rgb(var(--v-theme-primary)); }
.vtkweb-colormap-dropdown-chevron { opacity: .65; font-size: 14px; }
.vtkweb-colormap-dropdown-menu {
    position: absolute; z-index: 1000; top: calc(100% + 3px); left: 0;
    width: max(100%, 240px); max-height: 340px; overflow-y: auto;
    overscroll-behavior-y: contain; /* Do not scroll the inspector at menu edges. */
    padding: 4px; border-radius: 6px;
    background: rgb(var(--v-theme-surface));
    box-shadow: 0 5px 18px rgba(0,0,0,.30);
    border: 1px solid rgba(var(--v-theme-on-surface), .15);
}
.vtkweb-colormap-dropdown-option {
    display: flex; align-items: center; width: 100%; min-height: 34px;
    padding: 4px 10px; margin: 1px 0; border: 0; border-radius: 3px;
    cursor: pointer; text-align: left; font: inherit;
    background-size: 100% 100%; background-repeat: no-repeat;
    background-position: center;
}
.vtkweb-colormap-dropdown-option:hover,
.vtkweb-colormap-dropdown-option:focus-visible { outline: 2px solid rgb(var(--v-theme-primary)); outline-offset: -2px; }
.vtkweb-colormap-dropdown-option-label {
    color: #fff; font-size: 13px; font-weight: 600;
    text-shadow: 0 1px 3px #000, 0 0 5px #000, 1px 0 2px #000;
}




/* Combined opacity / color transfer-function editor. */
.vtkweb-tf-color-bar { position: relative; height: 27px; margin: 8px 9px 12px; border: 1px solid rgba(128,128,128,.6); border-radius: 3px; cursor: crosshair; touch-action: none; }
.vtkweb-tf-color-handle { position: absolute; top: 50%; width: 14px; height: 14px; transform: translate(-50%, -50%); border-radius: 50%; border: 2px solid white; box-shadow: 0 0 0 1px #333, 0 1px 4px #3339; cursor: grab; touch-action: none; }
.vtkweb-tf-color-handle:active { cursor: grabbing; }
.vtkweb-tf-color-picker-row { display: flex; align-items: center; gap: 8px; margin: 5px 0; }
.vtkweb-tf-native-color-picker { width: 40px; height: 26px; padding: 1px; cursor: pointer; }
.vtkweb-tf-toolbar { display: flex; align-items: center; gap: 6px; margin: 8px 0; }

/* Native color input lives inside the disc; clicking the disc opens it directly. */
.vtkweb-tf-handle-color-input {
    position: absolute;
    width: 1px;
    height: 1px;
    opacity: 0;
    pointer-events: none;
    padding: 0;
    border: 0;
}


.vtkweb-tf-color-preview { cursor: crosshair; touch-action: none; }
.vtkweb-tf-color-preview .vtkweb-tf-color-handle { cursor: grab; pointer-events: auto; }

/* Compact transfer-function toolbar above the combined editor. */
.vtkweb-tf-toolbar { display: flex; align-items: center; gap: 5px; margin: 9px 0 5px; position: relative; z-index: 5; }
.vtkweb-tf-tool-button { display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; padding: 0; border: 1px solid rgba(128,128,128,.35); border-radius: 5px; background: transparent; color: inherit; cursor: pointer; list-style: none; }
.vtkweb-tf-tool-button::-webkit-details-marker { display: none; }
.vtkweb-tf-tool-button:hover { background: rgba(128,128,128,.13); }
.vtkweb-tf-tool-menu { position: relative; }
.vtkweb-tf-tool-panel { position: absolute; left: 0; top: 36px; min-width: 250px; padding: 12px; background: rgb(var(--v-theme-surface)); border: 1px solid rgba(128,128,128,.35); border-radius: 6px; box-shadow: 0 5px 18px #0004; z-index: 20; }
.vtkweb-tf-preset-menu .vtkweb-colormap-dropdown-menu { left: 0; top: 36px; width: 260px; }
.vtkweb-tf-range-row { display: flex; align-items: center; gap: 6px; margin: 5px 0; }
.vtkweb-tf-range-label { min-width: 46px; font-size: 12px; }
.vtkweb-tf-range-row .vtkweb-range-input { width: 78px; min-width: 0; }
.vtkweb-tf-color-bar { margin-top: 2px; }

/* SVG color editor: gradient and handles read the same Vue state. */
.vtkweb-tf-color-svg { display: block; width: calc(100% - 18px); height: 30px; margin: 2px 9px 12px; overflow: visible; touch-action: none; cursor: crosshair; }
.vtkweb-tf-color-svg-handle { stroke: white; stroke-width: 2px; cursor: grab; pointer-events: all; vector-effect: non-scaling-stroke; }
.vtkweb-tf-color-svg-handle:active { cursor: grabbing; }

"""