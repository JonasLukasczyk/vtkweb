# VPT Henyey-Greenstein anisotropy checkpoint

- Adds VPT-only `vpt_anisotropy` control, -0.9 to +0.9, default 0.
- Evaluates the normalized HG phase density for HDRI next-event estimation.
- Samples continuing scattering directions from HG about the incoming ray direction.
- Uses `dot(ray.d, ds.d)` for the scattering cosine: `ds.d` points from collision toward the light, while `ray.d` points along incoming propagation. This is equivalent to the angle between incident propagation and outgoing scattered propagation for the camera-side NEE connection.
- Retains Bernoulli albedo, TF RGB throughput, custom spatial delta tracking and explicit RNG, and depth 1-4.
- `g=0` restores isotropic phase evaluation and uniform sphere sampling; RNG-to-direction mapping differs from the previous isotropic implementation but has the same distribution.
- The VPT integrator cache key includes `g`; DVR is unchanged.

## Tests

1. At depth 1 and 4, g=0 should statistically match the preceding checkpoint (not pixel-identical).
2. Compare g=-0.7, 0, +0.7 with directional HDRI features. Expect significant angular response changes.
3. Albedo=0 must retain absorption-only behavior.
4. Verify finite results and convergence at g=+-0.9.

CUDA/Mitsuba runtime execution has not been performed in this environment.
