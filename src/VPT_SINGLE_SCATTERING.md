# Custom VPT single-scattering checkpoint

- Keeps the explicit per-ray PCG state, spatial majorants, and 3D DDA camera-ray tracking.
- At the first accepted real collision, uses `scattering_albedo` as the probability of scattering rather than absorption.
- Scattering samples the Mitsuba environment emitter's *direction distribution* (HDRI importance sampling); it does **not** use Mitsuba's volumetric integrator.
- Evaluates an isotropic phase function, `1/(4*pi)`.
- Estimates shadow-ray medium transmittance with custom brick-DDA ratio tracking using explicit RNG state, and checks surface occlusion with `scene.ray_test`.
- Includes direct single scattering only; multiple scattering and MIS with phase sampling are not implemented.
- Scattering currently supports **one VPT volume**. Multiple overlapping media would need joint extinction and shadow traversal.
- Albedo zero bypasses the scattering path and preserves the existing absorption-only estimator.
- Changing albedo invalidates the integrator cache.

## Important limitations and validation

This checkpoint passed Python compilation and archive checks but **has not been run on CUDA/Mitsuba**. The `env.sample_direction` API and vectorized ray construction must be exercised on the user's `cuda_ad_rgb` environment. The first smoke test should use a low-extinction volume, HDRI, albedo 0 then 0.5. Confirm the albedo-zero image matches the prior checkpoint. At nonzero albedo, check that the volume becomes illuminated and that shadowing changes with density. The new shadow tracking uses an independent PCG stream derived from the camera-path state; it is not yet a separately seeded independent stream per light sample.

The 100,000-iteration guards remain in camera and shadow tracking. They fail loudly instead of silently returning biased results.
