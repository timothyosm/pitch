import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import time

# Constants
NUM_PARTICLES = 2000000
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Vertex shader
vertex_shader_code = """
#version 430

layout(location = 0) in vec3 in_position;

void main() {
    gl_Position = vec4(in_position, 1.0);
    gl_PointSize = 1.0;
}
"""

# Fragment shader
fragment_shader_code = """
#version 430

out vec4 fragColor;

void main() {
    fragColor = vec4(1, 1, 1, 1);
}
"""

# Compute shader
compute_shader_code = """
#version 430

layout(local_size_x = 64) in;

struct Particle {
    vec3 position;
    vec3 velocity;
};

layout(std430, binding = 0) buffer ssboParticlePositions {
    vec3 positions[];
};

layout(std430, binding = 1) buffer ssboParticleVelocities {
    vec3 velocities[];
};

uniform float time;

// Random number generator functions
uint pcg(uint n) {
    uint state = n * 747796405u + 2891336453u;
    state = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (state >> 22u) ^ state;
}

vec3 rand33(vec3 f) {
    uint x = floatBitsToUint(f.x);
    uint y = floatBitsToUint(f.y);
    uint z = floatBitsToUint(f.z);
    return vec3(pcg(x), pcg(y), pcg(z)) / float(0xffffffff);
}

// Perlin noise function (simplified for brevity)
float perlinNoise3(vec3 P) {
    return fract(sin(dot(P, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index >= positions.length()) return;

    vec3 randomOffset = rand33(positions[index] * 1024.0 + vec3(gl_GlobalInvocationID.x)) * 0.2 - 0.1;
    vec3 offset = vec3(time * 5.0, 0.0, 0.0) + randomOffset;

    velocities[index].x += perlinNoise3(positions[index] * (3.0 + cos(time)) + offset) * 0.2;
    velocities[index].y += perlinNoise3(positions[index] * 2.0 + 10.0 - offset) * 0.2;
    velocities[index] *= 0.9;
    velocities[index] = clamp(velocities[index], vec3(-1.0), vec3(1.0));

    positions[index] += velocities[index] * 0.01 + randomOffset * 0.05;

    for (int i = 0; i < 3; i++) {
        if (positions[index][i] < -1.1) {
            positions[index][i] = 1.0;
            velocities[index][i] *= -1.0;
        }
        if (positions[index][i] > 1.1) {
            positions[index][i] = -1.0;
            velocities[index][i] *= -1.0;
        }
    }
}
"""

def main():
    # Initialize GLFW
    if not glfw.init():
        return

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Particle System", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # Set up OpenGL options
    glEnable(GL_PROGRAM_POINT_SIZE)
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

    # Generate particle data
    positions = (np.random.rand(NUM_PARTICLES, 3).astype(np.float32) - 0.5) * 2.0
    velocities = (np.random.rand(NUM_PARTICLES, 3).astype(np.float32) - 0.5) * 0.2

    # Create buffers
    position_buffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, position_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, positions.nbytes, positions, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, position_buffer)

    velocity_buffer = glGenBuffers(1)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, velocity_buffer)
    glBufferData(GL_SHADER_STORAGE_BUFFER, velocities.nbytes, velocities, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, velocity_buffer)

    # Compile shaders
    vertex_shader = compileShader(vertex_shader_code, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_shader_code, GL_FRAGMENT_SHADER)
    render_program = compileProgram(vertex_shader, fragment_shader)

    compute_shader = compileShader(compute_shader_code, GL_COMPUTE_SHADER)
    compute_program = glCreateProgram()
    glAttachShader(compute_program, compute_shader)
    glLinkProgram(compute_program)
    if glGetProgramiv(compute_program, GL_LINK_STATUS) != GL_TRUE:
        print(glGetProgramInfoLog(compute_program))
        return

    # Set up VAO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, position_buffer)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    # Main loop
    start_time = time.time()
    while not glfw.window_should_close(window):
        # Compute shader pass
        glUseProgram(compute_program)
        current_time = time.time() - start_time
        time_loc = glGetUniformLocation(compute_program, "time")
        glUniform1f(time_loc, current_time)
        glDispatchCompute(NUM_PARTICLES // 64 + 1, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # Render pass
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(render_program)
        glBindVertexArray(vao)
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES)

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    # Clean up
    glDeleteBuffers(1, [position_buffer])
    glDeleteBuffers(1, [velocity_buffer])
    glDeleteVertexArrays(1, [vao])
    glDeleteProgram(render_program)
    glDeleteProgram(compute_program)

    glfw.terminate()

if __name__ == "__main__":
    main()
