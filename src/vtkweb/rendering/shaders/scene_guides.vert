#version 450
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_color;
layout(binding = 0) uniform Camera { mat4 mvp; } camera;
layout(location = 0) out vec4 color;
void main() {
    gl_Position = camera.mvp * vec4(in_position, 1.0);
    color = in_color;
}
