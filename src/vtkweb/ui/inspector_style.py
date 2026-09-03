INSPECTOR_STYLE = """
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

.vtkweb-select-control {
    flex: 1 1 auto;
    min-width: 0;
    width: 0;
}

.vtkweb-select-control .v-input__control,
.vtkweb-select-control .v-field {
    min-height: 26px !important;
    height: 26px !important;
}

.vtkweb-select-control .v-field {
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.vtkweb-select-control .v-field__outline {
    display: none !important;
}

.vtkweb-select-control .v-field__input {
    min-height: 26px !important;
    height: 26px !important;
    padding: 0 4px !important;
    font-size: 12px;
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

"""
