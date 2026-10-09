# VPT transfer-function scattering tint

At a real scattering event, the existing color TF is sampled at the scalar value of the collision and multiplies the single-scattering HDRI contribution. The color mapping uses its own scalar range and a 1024-entry per-channel lookup table. The color mapping is part of the VPT integrator cache key.

Extinction majorants, the opacity TF, background transmittance, and DVR are unchanged. Albedo zero remains absorption-only. Color values are used as supplied by the existing TF; this checkpoint does not introduce a new color-space conversion.

CUDA execution and radiometric validation have not been performed.
