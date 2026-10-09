# Custom VPT multiple-scattering checkpoint

- Adds VPT-only `vpt_max_depth` control (1 to 4, default 1).
- At each real scattering collision, accumulates direct HDRI next-event estimation using the existing spatial-majorant shadow ratio tracker.
- Samples an isotropic direction and traces a new ray from the collision for the next bounce.
- TF RGB is multiplied into path throughput at every scattering event. Albedo remains a Bernoulli event choice, so it is not multiplied again.
- Explicit PCG RNG state survives every DDA loop and every scattering bounce. Null collisions do not consume depth.
- Depth 1 remains the default for backwards compatibility. DVR is unchanged.

## Limitations

- Only one VPT volume is supported by this checkpoint.
- Secondary paths currently contribute through next-event estimation only; they do not evaluate visible surfaces or an escaped HDRI direction separately. Direct HDRI illumination is estimated at each scatter.
- No Russian roulette is needed for the bounded 1–4-scatter prototype. No phase/environment MIS yet.
- Source compilation was checked, but no CUDA render was performed here. Please test depth 1 first, then 2 and 4.
