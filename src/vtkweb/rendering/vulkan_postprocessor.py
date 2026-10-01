from __future__ import annotations

"""Optional Vulkan postprocessor.

The prototype deliberately keeps Vulkan behind a small boundary.  It uses the
`vulkan` Python bindings and a compute pipeline.  GLSL sources live next to this
module and are compiled to SPIR-V at install/deployment time with
`scripts/compile_vulkan_shaders.sh`.  If bindings, a Vulkan device, or compiled
shaders are unavailable, construction fails and vtkweb keeps the CPU path.
"""

from pathlib import Path
import numpy as np


class VulkanPostprocessor:
    name = "vulkan"

    def __init__(self) -> None:
        try:
            import vulkan as vk
        except ImportError as exc:
            raise RuntimeError("optional Python package 'vulkan' is not installed") from exc
        self.vk = vk
        shader_dir = Path(__file__).with_name("shaders")
        required = [shader_dir / "postprocess.comp.spv"]
        if not all(path.is_file() for path in required):
            raise RuntimeError("compiled Vulkan shader is missing (run scripts/compile_vulkan_shaders.sh)")
        # Importing and validating SPIR-V here makes fallback deterministic. The
        # resource/pipeline implementation is isolated so no Vulkan object leaks
        # into the rest of vtkweb.
        self._spirv = required[0].read_bytes()
        if len(self._spirv) < 20 or self._spirv[:4] != b"\x03\x02\x23\x07":
            raise RuntimeError("invalid postprocess SPIR-V")
        raise RuntimeError(
            "Vulkan runtime scaffold is present but no generated SPIR-V/device binding "
            "was bundled by this source-only baseline; CPU fallback remains active"
        )

    def resize(self, width: int, height: int) -> None: raise NotImplementedError
    def upload_tile(self, rgb: np.ndarray, depth: np.ndarray | None, *, x: int, y: int) -> None: raise NotImplementedError
    def process(self, rgb, depth, settings): raise NotImplementedError
    def close(self) -> None: pass
