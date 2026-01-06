import math
from ctypes import Structure, c_float, c_uint, sizeof
from typing import ClassVar

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_BUFFER_BIT,
    GL_COMPUTE_SHADER,
    GL_DRAW_FRAMEBUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_MAJOR_VERSION,
    GL_MINOR_VERSION,
    GL_R32UI,
    GL_READ_WRITE,
    GL_RED_INTEGER,
    GL_RGB,
    GL_RGBA,
    GL_RGBA16F,
    GL_SHADER_IMAGE_ACCESS_BARRIER_BIT,
    GL_SHADER_STORAGE_BARRIER_BIT,
    GL_SHADER_STORAGE_BUFFER,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE1,
    GL_TEXTURE_2D,
    GL_TEXTURE_2D_ARRAY,
    GL_UNSIGNED_INT,
    GL_WRITE_ONLY,
    glActiveTexture,
    glBindBuffer,
    glBindBufferBase,
    glBindFramebuffer,
    glBindImageTexture,
    glBindTexture,
    glBufferData,
    glBufferSubData,
    glClear,
    glClearTexImage,
    glDispatchCompute,
    glDrawBuffer,
    glFramebufferTexture,
    glGenBuffers,
    glGenFramebuffers,
    glGetIntegerv,
    glGetTexImage,
    glGetUniformLocation,
    glMemoryBarrier,
    glTexImage2D,
    glTexImage3D,
    glTexSubImage2D,
    glUniform1fv,
    glUniform1ui,
    glUniform2f,
    glUniform4f,
    glUniformMatrix3x2fv,
    glUseProgram,
    shaders,
)

from flames.renderer import compile_shader

from .common import HELPERS, FlameIterator, gen_texture_2d, gen_texture_2d_array


def allocate(weights, total):
    # sum(weights) == 1
    results = [0] * len(weights)

    minimum = 1 / (total + 1)
    contenders = [(i, weight) for i, weight in enumerate(weights) if weight > minimum]

    for i, _ in contenders:
        results[i] += 1

    def huntington_hill(weight, assigned):
        return weight / math.sqrt(assigned * (assigned + 1))

    for _ in range(total - len(contenders)):
        i, _ = max(contenders, key=lambda pair: huntington_hill(pair[1], results[pair[0]]))
        results[i] += 1

    return results


SHUFFLER_SHADER = """#version 450
layout(local_size_x = SHUFFLER_LOCAL_SIZE, local_size_y = SHUFFLER_LOCAL_SIZE) in;
struct Particle {
	vec2 pos; // 8+8
	vec4 col; // 16
	uvec4 randState; // 16
	uint mappedIndex; // 4+12
}; // 64 bytes

layout (std140, binding = 0) readonly buffer InputBuffer {
    uvec4 shufflerStateIn[LOCAL_UNITS];
    Particle partIn [];
};

layout (std140, binding = 1) buffer OutputBuffer {
    uvec4 shufflerStateOut[LOCAL_UNITS];
    Particle partOut [];
};

uint TausStep(uint z, int S1, int S2, int S3, uint M) {
    uint b = (((z << S1) ^ z) >> S2);
    return (((z & M) << S3) ^ b);
}

uint LCGStep(uint z, uint A, uint C) {
    return A*z + C;
}
uint randInt(inout uvec4 state) {
    state.x = TausStep(state.x, 13, 19, 12, 4294967294);
    state.y = TausStep(state.y, 2, 25, 4, 4294967288);
    state.z = TausStep(state.z, 3, 11, 17, 4294967280);
    state.w = LCGStep(state.w, 1664525, 1013904223);

    return state.x ^ state.y ^ state.z ^ state.w;
}
float randFloat(inout uvec4 state) {
    state.x = TausStep(state.x, 13, 19, 12, 4294967294);
    state.y = TausStep(state.y, 2, 25, 4, 4294967288);
    state.z = TausStep(state.z, 3, 11, 17, 4294967280);
    state.w = LCGStep(state.w, 1664525, 1013904223);

    return 2.3283064365387e-10 * float(state.x ^ state.y ^ state.z ^ state.w);
}


void main() {
    uint localIndex = gl_LocalInvocationIndex + gl_WorkGroupID.x * (SHUFFLER_LOCAL_SIZE*SHUFFLER_LOCAL_SIZE);

    uvec4 state = shufflerStateIn[localIndex];

    for (uint i = 0; i<BLOCKS-1; i++) {
        uint j = i + randInt(state) % (BLOCKS - i);
        //uint j = i + uint(randFloat(state) * (BLOCKS - i));
        //uint j = i + 0;

        uint final_i = i * LOCAL_UNITS + localIndex;
        uint final_j = j * LOCAL_UNITS + localIndex;

        uint temp = partOut[final_i].mappedIndex;
        partOut[final_i].mappedIndex = partOut[final_j].mappedIndex;
        partOut[final_j].mappedIndex = temp;
    }

    //UNROLLED_LOOP

    shufflerStateOut[localIndex] = state;
}
"""

BASE_SHADER = """#version 450
#define PI 3.141592653589

#define RETURN(expr) result += expr;
#define RANDOM random(state)
#define RAND_VEC vec2(random(state), random(state))

layout(local_size_x = LOCAL_SIZE, local_size_y = LOCAL_SIZE) in;
struct Particle {
	vec2 pos; // 8+8
	vec4 col; // 16
	uvec4 randState; // 16
	uint mappedIndex; // 4+12
}; // 64 bytes

layout (std140, binding = 0) readonly buffer InputBuffer {
        uvec4 shufflerStateIn[LOCAL_SIZE*LOCAL_SIZE];
	Particle partIn [];
};

layout (std140, binding = 1) writeonly buffer OutputBuffer {
        uvec4 shufflerStateOut[LOCAL_SIZE*LOCAL_SIZE];
	Particle partOut [];
};

layout (r32ui, binding = 0) uniform uimage2D histogramOut;
layout (rgba16f, binding = 1) uniform writeonly image2D colourOut;

uniform uint tick;
uniform vec2 minBound;
uniform vec2 maxBound;
uniform uint computeOffset;

uniform vec4 funcColour;
uniform mat3x2 preTrans;
uniform mat3x2 postTrans;
#if N_PARAMS > 0
uniform float params[N_PARAMS];
#endif

ivec2 colourSize = imageSize(colourOut);
ivec2 histSize = imageSize(histogramOut);

vec2 colourConversion = colourSize / (maxBound - minBound);
vec2 histConversion = histSize / (maxBound - minBound);

uint TausStep(uint z, int S1, int S2, int S3, uint M) {
    uint b = (((z << S1) ^ z) >> S2);
    return (((z & M) << S3) ^ b);
}

uint LCGStep(uint z, uint A, uint C) {
    return A*z + C;
}

float random(inout uvec4 state) {
    state.x = TausStep(state.x, 13, 19, 12, 4294967294);
    state.y = TausStep(state.y, 2, 25, 4, 4294967288);
    state.z = TausStep(state.z, 3, 11, 17, 4294967280);
    state.w = LCGStep(state.w, 1664525, 1013904223);

    return 2.3283064365387e-10 * float(state.x ^ state.y ^ state.z ^ state.w);
}

float gaussian(inout uvec4 state) {
	float val = -2;
	val += random(state);
	val += random(state);
	val += random(state);
	val += random(state);
	return val;
}

vec2 evaluate(inout uvec4 state, vec2 pos) {
    vec2 coord = preTrans * vec3(pos, 1);

    FUNCTION_BODY

    return postTrans * vec3(result, 1);
}


void main() {
	uint particle = gl_LocalInvocationIndex + (gl_WorkGroupID.x + computeOffset) * (LOCAL_SIZE*LOCAL_SIZE);
        particle = partIn[particle].mappedIndex; // Just a bit of indirection...

	uvec4 state = partIn[particle].randState;

	vec2 pos = evaluate(state, partIn[particle].pos);

	vec4 colour = mix(partIn[particle].col, vec4(funcColour.rgb,1), funcColour.a);
	if (tick >= 20) {
		imageAtomicAdd(histogramOut, ivec2((pos - minBound) * histConversion), uint(1));
		imageStore(colourOut, ivec2((pos - minBound) * colourConversion), colour);
	}
	partOut[particle].randState = state;
	partOut[particle].pos = pos;
	partOut[particle].col = colour;
}
"""


class Function:
    def __init__(self, function_formats, local_size):
        def generate_names(prefix):
            i = 0
            while True:
                yield prefix + str(i)
                i += 1

        name_generator = generate_names('i')
        img_generator = generate_names('img')

        self.images = {}

        i = 0
        function_text = HELPERS + 'vec2 result = vec2(0);\n'
        for function_format in function_formats:
            in_names = []
            for name in function_format.params:
                in_names.append('params[{}]'.format(i))
                i += 1

            new_images = []
            for filename in function_format.images.values():
                name = next(img_generator)
                new_images.append(name)
                self.images[name] = filename

            function_text += function_format.instantiate(in_names, name_generator, new_images) + '\n'
        self.n_params = i

        shader_text = BASE_SHADER.replace('N_PARAMS', str(self.n_params))
        shader_text = shader_text.replace('FUNCTION_BODY', function_text)

        # print(shader_text)

        shader = compile_shader(shader_text, GL_COMPUTE_SHADER, defines={'LOCAL_SIZE': local_size})
        self.program = shaders.compileProgram(shader)

        self.pre_trans_uniform = glGetUniformLocation(self.program, 'preTrans')
        self.post_trans_uniform = glGetUniformLocation(self.program, 'postTrans')

        self.colour_uniform = glGetUniformLocation(self.program, 'funcColour')

        self.min_bound_uniform = glGetUniformLocation(self.program, 'minBound')
        self.max_bound_uniform = glGetUniformLocation(self.program, 'maxBound')

        self.tick_uniform = glGetUniformLocation(self.program, 'tick')
        self.compute_offset_uniform = glGetUniformLocation(self.program, 'computeOffset')

        if self.n_params > 0:
            self.params_uniform = glGetUniformLocation(self.program, 'params')
        else:
            self.params_uniform = None

        if self.n_params > 0:
            self.params = (self.n_params * c_float)()
        else:
            self.params = []

        self.pre_trans = (6 * c_float)(1, 0, 0, 1, 0, 0)
        self.post_trans = (6 * c_float)(1, 0, 0, 1, 0, 0)
        self.colour = (4 * c_float)(1, 1, 1, 0.5)

        self.compute_size = None

    def set_compute_settings(self, compute_offset, compute_size):
        self.compute_size = compute_size

        glUseProgram(self.program)
        glUniform1ui(self.compute_offset_uniform, compute_offset)

    def update_uniforms(self, min_bound, max_bound):
        glUseProgram(self.program)

        glUniform2f(self.min_bound_uniform, min_bound[0], min_bound[1])
        glUniform2f(self.max_bound_uniform, max_bound[0], max_bound[1])

        glUniformMatrix3x2fv(self.pre_trans_uniform, 1, False, self.pre_trans)
        glUniformMatrix3x2fv(self.post_trans_uniform, 1, False, self.post_trans)
        glUniform4f(self.colour_uniform, self.colour[0], self.colour[1], self.colour[2], self.colour[3])

        if self.params_uniform is not None:
            glUniform1fv(self.params_uniform, self.n_params, self.params)

    def run_compute(self, tick, block_size):
        assert self.compute_size is not None

        glUseProgram(self.program)
        glUniform1ui(self.tick_uniform, tick)
        glDispatchCompute(self.compute_size * block_size, 1, 1)


class Particle(Structure):
    _fields_: ClassVar = [
        ('pos', c_float * 2),
        ('pad1', c_float * 2),
        ('col', c_float * 4),
        ('randState', c_uint * 4),
        ('mappedIndex', c_uint),
        ('pad2', c_float * 3),
    ]


def create_shuffler(blocks, local_units, shuffler_local_size):
    shader = compile_shader(
        SHUFFLER_SHADER,
        GL_COMPUTE_SHADER,
        defines={'LOCAL_UNITS': local_units, 'SHUFFLER_LOCAL_SIZE': shuffler_local_size, 'BLOCKS': blocks},
    )
    return shaders.compileProgram(shader)


class ShufflerFlameIterator(FlameIterator):
    def __init__(self, size: tuple[int, int], supersamples: int):
        super().__init__(size, supersamples)
        self.uses_markov_chain = False

        self.blocks = 1024
        self.local_size = 16
        self.block_size = 1

        self.local_units = self.local_size**2 * self.block_size
        self.particles = self.local_units * self.blocks

        self.tick = 0
        # self.max_iterations = 10_000_000_000 // self.particles

        glActiveTexture(GL_TEXTURE0)
        self.histogram = gen_texture_2d()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, *self.size, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, None)
        glBindImageTexture(0, self.histogram, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI)

        glActiveTexture(GL_TEXTURE1)
        self.colour = gen_texture_2d_array()
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA16F, *self.size, self.supersamples, 0, GL_RGBA, GL_FLOAT, None)
        glBindImageTexture(1, self.colour, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)

        self.buffers = glGenBuffers(2).tolist()
        buf_size = (sizeof(Particle) * self.blocks + 16) * self.local_units
        for buffer in self.buffers:
            glBindBuffer(GL_ARRAY_BUFFER, buffer)
            glBufferData(GL_ARRAY_BUFFER, buf_size, None, GL_STATIC_DRAW)

        shuffler_local_size = 4
        self.shuffler_work_size = self.local_units // (shuffler_local_size**2)
        self.shuffler_program = create_shuffler(self.blocks, self.local_units, shuffler_local_size)

        # Don't have glClearTexImage
        self.clear_mode = (glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION)) < (4, 4)
        if self.clear_mode:
            self.empty_histogram = np.zeros(self.size[0] * self.size[1], np.uint32)
            self.fbo = int(glGenFramebuffers(1))

    def update(self):
        buffer = bool(self.tick % 2)
        self.tick += 1

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.buffers[not buffer])  # in
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.buffers[buffer])  # out

        glUseProgram(self.shuffler_program)
        glDispatchCompute(self.shuffler_work_size, 1, 1)

        for function in self.functions:
            function.run_compute(self.tick, self.block_size)

        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT)

    def clear(self):
        if self.clear_mode:
            glBindTexture(GL_TEXTURE_2D, self.histogram)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, *self.size, GL_RED_INTEGER, GL_UNSIGNED_INT, self.empty_histogram)

            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.fbo)
            glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self.colour, 0)
            glDrawBuffer(GL_COLOR_ATTACHMENT0)
            glClear(GL_COLOR_BUFFER_BIT)

            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
            glBindTexture(GL_TEXTURE_2D, 0)
        else:
            glClearTexImage(self.colour, 0, GL_RGBA, GL_FLOAT, c_float(0))
            glClearTexImage(self.histogram, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, c_uint(0))

    def get_bounds(self):
        bounds = np.array([[-1, -1], [1, 1]], np.float32) * self.radius
        bounds[:, 0] *= self.size[0] / self.size[1]
        bounds += self.centre

        return bounds

    def reset(self):
        self.clear()

        self.tick = 0

        if len(self.functions) != 0:
            try:
                allocation = allocate(list(self.functions.values()), self.blocks)
            except ZeroDivisionError:
                return  # Have invalid probabilities

            min_bound, max_bound = self.get_bounds()
            i = 0
            for amount, function in zip(allocation, self.functions):
                function.set_compute_settings(i, amount)
                i += amount

                function.update_uniforms(min_bound, max_bound)

    def reset_all(self):
        self.reset()

        min_bound, max_bound = self.get_bounds()

        positions = np.random.random((self.particles, 2)).astype(np.float32)
        positions *= max_bound - min_bound
        positions += min_bound

        particle_data = (Particle * self.particles)()
        rand_states = np.random.randint(1 << 32, size=(len(positions), 4), dtype=np.uint32)

        for i, (entry, position, rand_state) in enumerate(zip(particle_data, positions, rand_states)):
            entry.pos[0] = position[0]
            entry.pos[1] = position[1]
            # entry.col should've already been initialised to (0,0,0,0)

            entry.randState[0] = rand_state[0]
            entry.randState[1] = rand_state[1]
            entry.randState[2] = rand_state[2]
            entry.randState[3] = rand_state[3]

            entry.mappedIndex = i

        shuffle_data = np.random.bytes(16 * self.local_units)

        for buffer in self.buffers:
            glBindBuffer(GL_ARRAY_BUFFER, buffer)
            glBufferSubData(GL_ARRAY_BUFFER, 0, len(shuffle_data), shuffle_data)
            glBufferSubData(GL_ARRAY_BUFFER, len(shuffle_data), sizeof(particle_data), particle_data)

    def read_textures(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.histogram)
        pixels = glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT)
        hist = np.frombuffer(pixels, np.uint32).reshape(self.size[::-1])

        glBindTexture(GL_TEXTURE_2D, self.colour)
        pixels = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT)
        colour = np.frombuffer(pixels, np.float32).reshape((*(self.size * self.supersamples)[::-1], 3))

        glBindTexture(GL_TEXTURE_2D, 0)

        return hist, colour

    def create_function(self, function_formats):
        func = Function(function_formats, self.local_size)
        self.functions[func] = 0.0
        return func

    def set_use_markov_chain(self, enabled):
        if enabled == self.uses_markov_chain:
            return

        if not enabled:
            raise NotImplementedError

        self.uses_markov_chain = enabled
