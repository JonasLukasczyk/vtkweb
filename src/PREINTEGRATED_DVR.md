

## Per-pixel phase jitter and cached shadow prefix correction

Progressive DVR now draws one uniform phase value from the Mitsuba sampler per camera ray/pixel and applies the existing golden-ratio frame rotation modulo one. This turns coherent march bands into spatial noise while preserving progressive phase coverage. The same per-ray phase is reused by live secondary marches.

Cached axis-aligned shadow textures store cumulative optical depth at scalar grid planes. They are no longer trilinearly sampled along the light axis at arbitrary points, which incorrectly linearized a generally nonlinear partial-cell optical-depth function near sharp transfer-function boundaries. A cached query now integrates the partial cell from the query point to the next grid plane toward the light using the same pre-integration table, then adds the cached prefix value at that plane. Transverse texture interpolation remains enabled.
