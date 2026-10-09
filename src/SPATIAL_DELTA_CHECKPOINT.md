# Spatial delta tracking checkpoint

- Replaces global-majorant stochastic absorption with 8-voxel bricks and local conservative extinction bounds.
- Bounds include a one-voxel scalar halo for trilinear interpolation and all opacity transfer-function knots within each brick's scalar interval.
- Per-ray traversal advances to the next brick boundary; candidate free flights use the local majorant.
- No collision cap or deterministic fallback is used in delta mode. Deterministic preintegration remains selectable.
- Current implementation uses a small world-space boundary epsilon (1e-6); very tiny world-coordinate volumes may require a scale-relative DDA boundary policy.
- The brick cache is built on the CPU when the integrator is reconstructed. GPU compilation and visual convergence were not tested in this environment.
- This remains absorption-only: scattering albedo does not yet affect the image.
