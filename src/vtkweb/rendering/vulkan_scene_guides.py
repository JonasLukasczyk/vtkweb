from __future__ import annotations

"""Persistent off-screen Vulkan line renderer for viewport scene guides.

This renderer owns its Vulkan instance/device/queue and persistent off-screen
resources. Camera motion only updates a small uniform buffer; grid/axis geometry
is rebuilt only when guide settings change, and framebuffer resources only on
resize. The result is copied to a host-visible RGBA buffer and alpha-composited
onto the already post-processed RGB image. GPU->CPU readback is intentional for
this prototype because the existing H.264 encoder is CPU/PyAV.
"""

import ctypes
import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np

from vtkweb.rendering.scene_guides import (
    SceneCamera,
    SceneGuideSettings,
    build_guide_vertices,
    view_projection,
)


class VulkanSceneGuideRenderer:
    name = "vulkan"

    def __init__(self) -> None:
        try:
            import vulkan as vk
        except ImportError as exc:
            raise RuntimeError("optional Python package 'vulkan' is not installed") from exc
        self.vk = vk
        self.width = 0
        self.height = 0
        self._geometry_key = None
        self._vertex_count = 0
        self._closed = False
        self._create_device()
        self._load_shaders()
        self._create_command_pool()
        self._create_descriptor_layout()
        self._create_pipeline_layout()
        self._uniform_buffer, self._uniform_memory = self._create_buffer(
            64,
            vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        self._vertex_buffer = None
        self._vertex_memory = None
        self._image = self._image_memory = self._image_view = None
        self._readback_buffer = self._readback_memory = None
        self._render_pass = self._framebuffer = self._pipeline = None
        self._descriptor_pool = self._descriptor_set = None
        self._command_buffer = None

    # ------------------------------------------------------------------ setup
    def _create_device(self) -> None:
        vk = self.vk
        app = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="vtkweb-scene-guides",
            applicationVersion=1,
            pEngineName="vtkweb",
            engineVersion=1,
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        self.instance = vk.vkCreateInstance(
            vk.VkInstanceCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                pApplicationInfo=app,
            ),
            None,
        )
        physical_devices = vk.vkEnumeratePhysicalDevices(self.instance)
        if not physical_devices:
            raise RuntimeError("no Vulkan physical device found")
        selected = None
        queue_family = None
        for physical in physical_devices:
            props = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical)
            for index, prop in enumerate(props):
                if prop.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                    selected, queue_family = physical, index
                    break
            if selected is not None:
                break
        if selected is None:
            raise RuntimeError("no Vulkan graphics queue family found")
        self.physical_device = selected
        self.queue_family = int(queue_family)
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.queue_family,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        self.device = vk.vkCreateDevice(
            self.physical_device,
            vk.VkDeviceCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                queueCreateInfoCount=1,
                pQueueCreateInfos=[queue_info],
            ),
            None,
        )
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family, 0)
        self.memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)

    def _load_shaders(self) -> None:
        shader_dir = Path(__file__).with_name("shaders")
        sources = [shader_dir / "scene_guides.vert", shader_dir / "scene_guides.frag"]
        outputs = [Path(str(path) + ".spv") for path in sources]
        if not all(path.is_file() for path in outputs):
            glslc = shutil.which("glslc")
            if glslc:
                for source, output in zip(sources, outputs):
                    subprocess.run([glslc, str(source), "-o", str(output)], check=True)
        if not all(path.is_file() for path in outputs):
            raise RuntimeError(
                "scene-guide SPIR-V is missing; install glslc and run "
                "scripts/compile_vulkan_shaders.sh"
            )
        self._vert_spirv = outputs[0].read_bytes()
        self._frag_spirv = outputs[1].read_bytes()

    def _create_command_pool(self) -> None:
        vk = self.vk
        self.command_pool = vk.vkCreateCommandPool(
            self.device,
            vk.VkCommandPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                queueFamilyIndex=self.queue_family,
            ),
            None,
        )

    def _create_descriptor_layout(self) -> None:
        vk = self.vk
        binding = vk.VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1,
            stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
        )
        self.descriptor_layout = vk.vkCreateDescriptorSetLayout(
            self.device,
            vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=1,
                pBindings=[binding],
            ),
            None,
        )

    def _create_pipeline_layout(self) -> None:
        vk = self.vk
        self.pipeline_layout = vk.vkCreatePipelineLayout(
            self.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_layout],
            ),
            None,
        )

    def _memory_type(self, bits: int, flags: int) -> int:
        for index in range(self.memory_properties.memoryTypeCount):
            if bits & (1 << index):
                memory_type = self.memory_properties.memoryTypes[index]
                if (memory_type.propertyFlags & flags) == flags:
                    return index
        raise RuntimeError(f"no Vulkan memory type for flags 0x{flags:x}")

    def _create_buffer(self, size: int, usage: int, memory_flags: int):
        vk = self.vk
        buffer = vk.vkCreateBuffer(
            self.device,
            vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=max(1, int(size)),
                usage=usage,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        memory = vk.vkAllocateMemory(
            self.device,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=requirements.size,
                memoryTypeIndex=self._memory_type(requirements.memoryTypeBits, memory_flags),
            ),
            None,
        )
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        return buffer, memory

    def _write_memory(self, memory, data: bytes) -> None:
        vk = self.vk
        pointer = vk.vkMapMemory(self.device, memory, 0, len(data), 0)
        if hasattr(vk, "ffi"):
            vk.ffi.memmove(pointer, data, len(data))
        else:
            ctypes.memmove(int(pointer), data, len(data))
        vk.vkUnmapMemory(self.device, memory)

    # -------------------------------------------------------------- framebuffer
    def resize(self, width: int, height: int) -> None:
        width, height = max(1, int(width)), max(1, int(height))
        if (width, height) == (self.width, self.height):
            return
        self.vk.vkDeviceWaitIdle(self.device)
        self._destroy_frame_resources()
        self.width, self.height = width, height
        self._create_image()
        self._create_render_pass()
        self._create_pipeline()
        self._create_framebuffer()
        self._create_descriptor_set()
        self._create_command_buffer()
        self._readback_buffer, self._readback_memory = self._create_buffer(
            width * height * 4,
            self.vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            self.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | self.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

    def _create_image(self) -> None:
        vk = self.vk
        info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            extent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        self._image = vk.vkCreateImage(self.device, info, None)
        req = vk.vkGetImageMemoryRequirements(self.device, self._image)
        self._image_memory = vk.vkAllocateMemory(
            self.device,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=req.size,
                memoryTypeIndex=self._memory_type(req.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            ),
            None,
        )
        vk.vkBindImageMemory(self.device, self._image, self._image_memory, 0)
        self._image_view = vk.vkCreateImageView(
            self.device,
            vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=self._image,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=vk.VK_FORMAT_R8G8B8A8_UNORM,
                components=vk.VkComponentMapping(
                    r=vk.VK_COMPONENT_SWIZZLE_IDENTITY, g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    b=vk.VK_COMPONENT_SWIZZLE_IDENTITY, a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                ),
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
                ),
            ),
            None,
        )

    def _create_render_pass(self) -> None:
        vk = self.vk
        attachment = vk.VkAttachmentDescription(
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        )
        reference = vk.VkAttachmentReference(attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=1,
            pColorAttachments=[reference],
        )
        dependency = vk.VkSubpassDependency(
            srcSubpass=0,
            dstSubpass=vk.VK_SUBPASS_EXTERNAL,
            srcStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            dstStageMask=vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            srcAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
        )
        self._render_pass = vk.vkCreateRenderPass(
            self.device,
            vk.VkRenderPassCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                attachmentCount=1, pAttachments=[attachment],
                subpassCount=1, pSubpasses=[subpass],
                dependencyCount=1, pDependencies=[dependency],
            ),
            None,
        )

    def _shader_module(self, code: bytes):
        vk = self.vk
        return vk.vkCreateShaderModule(
            self.device,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(code),
                pCode=code,
            ),
            None,
        )

    def _create_pipeline(self) -> None:
        vk = self.vk
        vert = self._shader_module(self._vert_spirv)
        frag = self._shader_module(self._frag_spirv)
        try:
            stages = [
                vk.VkPipelineShaderStageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    stage=vk.VK_SHADER_STAGE_VERTEX_BIT, module=vert, pName="main"),
                vk.VkPipelineShaderStageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT, module=frag, pName="main"),
            ]
            binding = vk.VkVertexInputBindingDescription(binding=0, stride=28, inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX)
            attrs = [
                vk.VkVertexInputAttributeDescription(location=0, binding=0, format=vk.VK_FORMAT_R32G32B32_SFLOAT, offset=0),
                vk.VkVertexInputAttributeDescription(location=1, binding=0, format=vk.VK_FORMAT_R32G32B32A32_SFLOAT, offset=12),
            ]
            vertex_input = vk.VkPipelineVertexInputStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                vertexBindingDescriptionCount=1, pVertexBindingDescriptions=[binding],
                vertexAttributeDescriptionCount=2, pVertexAttributeDescriptions=attrs,
            )
            assembly = vk.VkPipelineInputAssemblyStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                topology=vk.VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                primitiveRestartEnable=vk.VK_FALSE,
            )
            viewport = vk.VkViewport(x=0.0, y=0.0, width=float(self.width), height=float(self.height), minDepth=0.0, maxDepth=1.0)
            scissor = vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0), extent=vk.VkExtent2D(width=self.width, height=self.height))
            viewport_state = vk.VkPipelineViewportStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                viewportCount=1, pViewports=[viewport], scissorCount=1, pScissors=[scissor],
            )
            raster = vk.VkPipelineRasterizationStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                depthClampEnable=vk.VK_FALSE, rasterizerDiscardEnable=vk.VK_FALSE,
                polygonMode=vk.VK_POLYGON_MODE_FILL, cullMode=vk.VK_CULL_MODE_NONE,
                frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE, depthBiasEnable=vk.VK_FALSE,
                lineWidth=1.0,
            )
            multisample = vk.VkPipelineMultisampleStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
            )
            blend_attachment = vk.VkPipelineColorBlendAttachmentState(
                blendEnable=vk.VK_FALSE,
                colorWriteMask=(vk.VK_COLOR_COMPONENT_R_BIT | vk.VK_COLOR_COMPONENT_G_BIT |
                                vk.VK_COLOR_COMPONENT_B_BIT | vk.VK_COLOR_COMPONENT_A_BIT),
            )
            blend = vk.VkPipelineColorBlendStateCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                attachmentCount=1, pAttachments=[blend_attachment],
            )
            result = vk.vkCreateGraphicsPipelines(
                self.device, vk.VK_NULL_HANDLE, 1,
                [vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=2, pStages=stages,
                    pVertexInputState=vertex_input, pInputAssemblyState=assembly,
                    pViewportState=viewport_state, pRasterizationState=raster,
                    pMultisampleState=multisample, pColorBlendState=blend,
                    layout=self.pipeline_layout, renderPass=self._render_pass, subpass=0,
                )],
                None,
            )
            self._pipeline = result[0] if isinstance(result, (list, tuple)) else result
        finally:
            vk.vkDestroyShaderModule(self.device, vert, None)
            vk.vkDestroyShaderModule(self.device, frag, None)

    def _create_framebuffer(self) -> None:
        vk = self.vk
        self._framebuffer = vk.vkCreateFramebuffer(
            self.device,
            vk.VkFramebufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=self._render_pass,
                attachmentCount=1, pAttachments=[self._image_view],
                width=self.width, height=self.height, layers=1,
            ),
            None,
        )

    def _create_descriptor_set(self) -> None:
        vk = self.vk
        self._descriptor_pool = vk.vkCreateDescriptorPool(
            self.device,
            vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=1,
                poolSizeCount=1,
                pPoolSizes=[vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1)],
            ),
            None,
        )
        allocated = vk.vkAllocateDescriptorSets(
            self.device,
            vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=self._descriptor_pool,
                descriptorSetCount=1,
                pSetLayouts=[self.descriptor_layout],
            ),
        )
        self._descriptor_set = allocated[0]
        buffer_info = vk.VkDescriptorBufferInfo(buffer=self._uniform_buffer, offset=0, range=64)
        write = vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self._descriptor_set, dstBinding=0, descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, pBufferInfo=[buffer_info],
        )
        vk.vkUpdateDescriptorSets(self.device, 1, [write], 0, None)

    def _create_command_buffer(self) -> None:
        buffers = self.vk.vkAllocateCommandBuffers(
            self.device,
            self.vk.VkCommandBufferAllocateInfo(
                sType=self.vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self.command_pool,
                level=self.vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )
        self._command_buffer = buffers[0]

    # --------------------------------------------------------------- per frame
    def _ensure_geometry(self, settings: SceneGuideSettings) -> None:
        key = hashlib.sha1(repr(settings).encode("utf-8")).hexdigest()
        if key == self._geometry_key:
            return
        vertices = build_guide_vertices(settings)
        data = vertices.tobytes(order="C")
        if self._vertex_buffer is not None:
            self.vk.vkDeviceWaitIdle(self.device)
            self.vk.vkDestroyBuffer(self.device, self._vertex_buffer, None)
            self.vk.vkFreeMemory(self.device, self._vertex_memory, None)
        self._vertex_buffer, self._vertex_memory = self._create_buffer(
            max(1, len(data)),
            self.vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            self.vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | self.vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        if data:
            self._write_memory(self._vertex_memory, data)
        self._vertex_count = len(vertices)
        self._geometry_key = key

    def _record_and_submit(self) -> None:
        vk = self.vk
        cmd = self._command_buffer
        vk.vkResetCommandBuffer(cmd, 0)
        vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO))
        clear = vk.VkClearValue(color=vk.VkClearColorValue(float32=[0.0, 0.0, 0.0, 0.0]))
        vk.vkCmdBeginRenderPass(
            cmd,
            vk.VkRenderPassBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=self._render_pass,
                framebuffer=self._framebuffer,
                renderArea=vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0), extent=vk.VkExtent2D(width=self.width, height=self.height)),
                clearValueCount=1, pClearValues=[clear],
            ),
            vk.VK_SUBPASS_CONTENTS_INLINE,
        )
        if self._vertex_count:
            vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self._pipeline)
            vk.vkCmdBindDescriptorSets(cmd, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline_layout, 0, 1, [self._descriptor_set], 0, None)
            vk.vkCmdBindVertexBuffers(cmd, 0, 1, [self._vertex_buffer], [0])
            vk.vkCmdDraw(cmd, self._vertex_count, 1, 0, 0)
        vk.vkCmdEndRenderPass(cmd)
        region = vk.VkBufferImageCopy(
            bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=0, baseArrayLayer=0, layerCount=1),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
        )
        vk.vkCmdCopyImageToBuffer(cmd, self._image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, self._readback_buffer, 1, [region])
        vk.vkEndCommandBuffer(cmd)
        vk.vkQueueSubmit(
            self.queue, 1,
            [vk.VkSubmitInfo(sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO, commandBufferCount=1, pCommandBuffers=[cmd])],
            vk.VK_NULL_HANDLE,
        )
        vk.vkQueueWaitIdle(self.queue)

    def _read_rgba(self) -> np.ndarray:
        vk = self.vk
        size = self.width * self.height * 4
        pointer = vk.vkMapMemory(self.device, self._readback_memory, 0, size, 0)
        if hasattr(vk, "ffi"):
            raw = bytes(vk.ffi.buffer(pointer, size))
        else:
            raw = ctypes.string_at(int(pointer), size)
        vk.vkUnmapMemory(self.device, self._readback_memory)
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4).copy()

    def render(self, rgb: np.ndarray, camera: SceneCamera, settings: SceneGuideSettings) -> np.ndarray:
        if not settings.enabled:
            return rgb
        height, width = rgb.shape[:2]
        self.resize(width, height)
        self._ensure_geometry(settings)
        # GLSL matrices are column-major by default. NumPy stores rows contiguously,
        # so transpose before upload to preserve the mathematical matrix.
        mvp = view_projection(camera, width, height).T.astype(np.float32, copy=False)
        self._write_memory(self._uniform_memory, mvp.tobytes(order="C"))
        self._record_and_submit()
        overlay = self._read_rgba().astype(np.float32)
        alpha = overlay[..., 3:4] / 255.0
        return np.clip(rgb.astype(np.float32) * (1.0 - alpha) + overlay[..., :3] * alpha, 0.0, 255.0).astype(np.uint8)

    # ---------------------------------------------------------------- cleanup
    def _destroy_frame_resources(self) -> None:
        vk = self.vk
        if self._command_buffer is not None:
            vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [self._command_buffer]); self._command_buffer = None
        if self._descriptor_pool is not None:
            vk.vkDestroyDescriptorPool(self.device, self._descriptor_pool, None); self._descriptor_pool = None
        if self._framebuffer is not None:
            vk.vkDestroyFramebuffer(self.device, self._framebuffer, None); self._framebuffer = None
        if self._pipeline is not None:
            vk.vkDestroyPipeline(self.device, self._pipeline, None); self._pipeline = None
        if self._render_pass is not None:
            vk.vkDestroyRenderPass(self.device, self._render_pass, None); self._render_pass = None
        if self._image_view is not None:
            vk.vkDestroyImageView(self.device, self._image_view, None); self._image_view = None
        if self._image is not None:
            vk.vkDestroyImage(self.device, self._image, None); self._image = None
        if self._image_memory is not None:
            vk.vkFreeMemory(self.device, self._image_memory, None); self._image_memory = None
        if self._readback_buffer is not None:
            vk.vkDestroyBuffer(self.device, self._readback_buffer, None); self._readback_buffer = None
        if self._readback_memory is not None:
            vk.vkFreeMemory(self.device, self._readback_memory, None); self._readback_memory = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        vk = self.vk
        vk.vkDeviceWaitIdle(self.device)
        self._destroy_frame_resources()
        if self._vertex_buffer is not None:
            vk.vkDestroyBuffer(self.device, self._vertex_buffer, None)
            vk.vkFreeMemory(self.device, self._vertex_memory, None)
        vk.vkDestroyBuffer(self.device, self._uniform_buffer, None)
        vk.vkFreeMemory(self.device, self._uniform_memory, None)
        vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
        vk.vkDestroyDescriptorSetLayout(self.device, self.descriptor_layout, None)
        vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        vk.vkDestroyDevice(self.device, None)
        vk.vkDestroyInstance(self.instance, None)
