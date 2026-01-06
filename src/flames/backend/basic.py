from ctypes import Structure, c_float, c_uint, pointer, sizeof
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
    glDeleteTextures,
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
    glUniform1f,
    glUniform1fv,
    glUniform1i,
    glUniform1ui,
    glUniform2f,
    glUniform4fv,
    glUniformMatrix3x2fv,
    glUseProgram,
    shaders,
)

from flames.renderer import compile_shader

from .common import HELPERS, FlameIterator, FunctionFormat, gen_texture_2d, gen_texture_2d_array, load_image

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
#ifdef USES_MARKOV_CHAIN
    uint prev; // 4+12
#endif
}; // 64 bytes

layout (std140, binding = 0) readonly buffer InputBuffer {
	Particle partIn [];
};

layout (std140, binding = 1) writeonly buffer OutputBuffer {
	Particle partOut [];
};

layout (r32ui, binding = 0) uniform uimage2D histogramOut;
layout (rgba16f, binding = 1) uniform writeonly image2DArray colourOut;

uniform uint tick;
uniform int supersample;
uniform vec2 minBound;
uniform vec2 maxBound;

PARAMS_UNIFORMS

#ifdef USES_MARKOV_CHAIN
uniform float cutoff[FUNC_COUNT][FUNC_COUNT];
#else
uniform float cutoff[FUNC_COUNT];
#endif

uniform vec4 colours[FUNC_COUNT];
uniform mat3x2 preTransforms[FUNC_COUNT];
uniform mat3x2 postTransforms[FUNC_COUNT];

ivec2 histSize = imageSize(histogramOut);

vec2 conversion = histSize / (maxBound - minBound);

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

FUNCTIONS

void main() {
	uint particle = gl_LocalInvocationIndex + gl_WorkGroupID.x * (LOCAL_SIZE*LOCAL_SIZE);
	uvec4 state = partIn[particle].randState;

    float accumulator = random(state);
    vec2 pos;
    vec4 funcColour;
#ifdef USES_MARKOV_CHAIN
    uint funcIndex;
#endif

    APPLY_FUNCTION

	vec4 colour = mix(partIn[particle].col, vec4(funcColour.rgb, 1), funcColour.a);
	if (tick >= 20) {
		imageAtomicAdd(histogramOut, ivec2((pos - minBound) * conversion), uint(1));
		imageStore(colourOut, ivec3((pos - minBound) * conversion, supersample), colour);
	}
	partOut[particle].randState = state;
	partOut[particle].pos = pos;
	partOut[particle].col = colour;
#ifdef USES_MARKOV_CHAIN
    partOut[particle].prev = funcIndex;
#endif
}
"""


class Function:
    def __init__(self, function_formats: list[FunctionFormat]):
        self.formats = function_formats
        self.params = [0] * sum(len(func.params) for func in self.formats)
        if len(self.params) != 0:
            self.params = (c_float * len(self.params))(*self.params)
        self.pre_trans = (c_float * 6)(1, 0, 0, 1, 0, 0)
        self.post_trans = (c_float * 6)(1, 0, 0, 1, 0, 0)
        self.colour = (c_float * 4)(1, 1, 1, 0.5)

    def generate_function(self, func_id: int, image_name_gen):
        def generate_names():
            i = 0
            while True:
                yield 'i' + str(i)
                i += 1

        name_generator = generate_names()

        images = {}
        i = 0
        body = ''
        for func_format in self.formats:
            names = []
            for _ in func_format.params:
                names.append(f'params{func_id}[{i}]')
                i += 1

            new_images = []
            for filename, name in zip(func_format.images.values(), image_name_gen):
                images[name] = filename
                new_images.append(name)
            body += func_format.instantiate(names, name_generator, new_images) + '\n'

        return (
            f"""vec2 evaluate{func_id}(inout uvec4 state, inout vec4 funcColour, vec2 pos) {{
const mat3x2 preTrans = preTransforms[{func_id}];
const mat3x2 postTrans = postTransforms[{func_id}];

vec2 coord = preTrans * vec3(pos, 1);
{HELPERS}
vec2 result = vec2(0);
{body}
return postTrans * vec3(result, 1);
}}
""",
            images,
        )


class Particle(Structure):
    _fields_: ClassVar = [
        ('pos', c_float * 2),
        ('pad1', c_float * 2),
        ('col', c_float * 4),
        ('randState', c_uint * 4),
        # ('pad2', c_float * 4),
    ]


class ParticleWithMarkov(Structure):
    _fields_: ClassVar = [
        ('pos', c_float * 2),
        ('pad1', c_float * 2),
        ('col', c_float * 4),
        ('randState', c_uint * 4),
        ('prev', c_uint),
        ('pad2', c_uint * 3),
    ]


class BasicFlameIterator(FlameIterator):
    def __init__(self, size: tuple[int, int], supersamples: int):
        super().__init__(size, supersamples)

        self.uses_markov_chain = False

        self.blocks = 1024
        self.local_size = 16
        self.block_size = 1

        self.local_units = self.local_size**2 * self.block_size
        self.particles = self.local_units * self.blocks

        self.tick = 0

        glActiveTexture(GL_TEXTURE0)
        self.histogram = gen_texture_2d()
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32UI, *self.size, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, None)
        glBindImageTexture(0, self.histogram, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI)

        glActiveTexture(GL_TEXTURE1)
        self.colour = gen_texture_2d_array()
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA16F, *self.size, self.supersamples, 0, GL_RGBA, GL_FLOAT, None)
        glBindImageTexture(1, self.colour, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA16F)

        self.create_buffers()

        self.textures = {}

        self.generate_program()

        # Don't have glClearTexImage
        self.clear_mode = (glGetIntegerv(GL_MAJOR_VERSION), glGetIntegerv(GL_MINOR_VERSION)) < (4, 4)
        if self.clear_mode:
            self.empty_histogram = np.zeros(self.size[0] * self.size[1], np.uint32)
            self.fbo = int(glGenFramebuffers(1))

    def update(self):
        if self.program is not None:
            buffer = bool(self.tick % 2)

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.buffers[not buffer])  # in
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.buffers[buffer])  # out

            glUseProgram(self.program)
            glUniform1ui(self.tick_uniform, self.tick)
            glUniform1i(self.supersample_uniform, self.tick % self.supersamples)
            glDispatchCompute(self.blocks, 1, 1)

            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT)

        self.tick += 1

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
            glClearTexImage(self.colour, 0, GL_RGBA, GL_FLOAT, pointer(c_float(0.0)))
            glClearTexImage(self.histogram, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, pointer(c_uint(0)))

    def get_bounds(self):
        bounds = np.array([[-1, -1], [1, 1]], np.float32) * self.radius
        bounds[:, 0] *= self.size[0] / self.size[1]
        bounds += self.centre

        return bounds

    def update_textures(self):
        glUseProgram(self.program)
        for name, texture in self.textures.items():
            glUniform1i(glGetUniformLocation(self.program, name), texture)

    def reset(self):
        self.clear()
        if self.prev_functions != list(self.functions.keys()):
            self.generate_program()

        self.tick = 0

        if len(self.functions) == 0:
            return

        glUseProgram(self.program)

        min_bound, max_bound = self.get_bounds()
        glUniform2f(self.min_bound_uniform, min_bound[0], min_bound[1])
        glUniform2f(self.max_bound_uniform, max_bound[0], max_bound[1])

        acc = 0
        for i, (param_uniform, (func, prob)) in enumerate(zip(self.parameter_uniforms, self.functions.items())):
            glUniformMatrix3x2fv(glGetUniformLocation(self.program, f'preTransforms[{i}]'), 1, False, func.pre_trans)
            glUniformMatrix3x2fv(glGetUniformLocation(self.program, f'postTransforms[{i}]'), 1, False, func.post_trans)
            glUniform4fv(glGetUniformLocation(self.program, f'colours[{i}]'), 1, func.colour)

            if self.uses_markov_chain:
                acc = 0
                for j, item in enumerate(prob):
                    acc += item
                    glUniform1f(glGetUniformLocation(self.program, f'cutoff[{i}][{j}]'), acc)
            else:
                acc += prob
                glUniform1f(glGetUniformLocation(self.program, f'cutoff[{i}]'), acc)

            if param_uniform is not None:
                glUniform1fv(param_uniform, len(func.params), func.params)

    def generate_program(self):
        glDeleteTextures(list(self.textures.values()))
        self.texture = {}

        self.prev_functions = list(self.functions.keys())

        if len(self.prev_functions) == 0:
            self.program = None
            self.tick_uniform = None
            # self.cutoffs_uniform = None
            self.parameter_uniforms = []
            # self.pre_trans_uniform = None
            # self.post_trans_uniform = None
            # self.colour_uniform = None
            self.min_bound_uniform = None
            self.max_bound_uniform = None
            return

        def generate_names():
            i = 0
            while True:
                yield 'img' + str(i)
                i += 1

        image_name_gen = generate_names()

        params_uniforms_text = '\n'.join(
            f'uniform float params{i}[{len(func.params)}];'
            for i, func in enumerate(self.prev_functions)
            if len(func.params) > 0
        )

        images = {}
        functions_text = '\n'
        for i, func in enumerate(self.prev_functions):
            text, new_images = func.generate_function(i, image_name_gen)
            images.update(new_images)
            functions_text += text + '\n\n'

        params_uniforms_text += '\n\n' + '\n'.join('uniform sampler2D {};'.format(name) for name in images)

        if self.uses_markov_chain:
            if len(self.prev_functions) > 1:
                apply_function_text = ''
                for i, func in enumerate(self.prev_functions[:-1]):
                    apply_function_text += f"""if (accumulator < cutoff[partIn[particle].prev][{i}]) {{
funcColour = colours[{i}];
funcIndex={i};
pos = evaluate{i}(state, funcColour, partIn[particle].pos);
}} else """
                i += 1
                apply_function_text += f"""{{
funcColour = colours[{i}];
funcIndex={i};
pos = evaluate{i}(state, funcColour, partIn[particle].pos);
}}"""
            else:
                apply_function_text = """funcColour = colours[0];
funcIndex=0;
pos = evaluate0(state, funcColour, partIn[particle].pos);
"""
        else:
            if len(self.prev_functions) > 1:
                apply_function_text = ''
                for i, func in enumerate(self.prev_functions[:-1]):
                    apply_function_text += f"""if (accumulator < cutoff[{i}]) {{
funcColour = colours[{i}];
pos = evaluate{i}(state, funcColour, partIn[particle].pos);
}} else """
                i += 1
                apply_function_text += f"""{{
funcColour = colours[{i}];
pos = evaluate{i}(state, funcColour, partIn[particle].pos);
}}"""
            else:
                apply_function_text = """funcColour = colours[0];
pos = evaluate0(state, funcColour, partIn[particle].pos);
"""

        text = BASE_SHADER
        text = text.replace('APPLY_FUNCTION', apply_function_text)
        text = text.replace('FUNCTIONS', functions_text)
        text = text.replace('PARAMS_UNIFORMS', params_uniforms_text)

        # print(text)
        shader = compile_shader(
            text,
            GL_COMPUTE_SHADER,
            {
                'LOCAL_SIZE': self.local_size,
                'FUNC_COUNT': len(self.prev_functions),
                'USES_MARKOV_CHAIN': self.uses_markov_chain,
            },
        )
        self.program = shaders.compileProgram(shader)

        self.tick_uniform = glGetUniformLocation(self.program, 'tick')
        # self.cutoffs_uniform = glGetUniformLocation(self.program, 'cutoff')
        # self.pre_trans_uniform = glGetUniformLocation(self.program, 'preTransforms')
        # self.post_trans_uniform = glGetUniformLocation(self.program, 'postTransforms')
        # self.colour_uniform = glGetUniformLocation(self.program, 'colours')
        self.min_bound_uniform = glGetUniformLocation(self.program, 'minBound')
        self.max_bound_uniform = glGetUniformLocation(self.program, 'maxBound')
        self.supersample_uniform = glGetUniformLocation(self.program, 'supersample')

        self.parameter_uniforms = []
        for i, func in enumerate(self.prev_functions):
            if len(func.params) == 0:
                self.parameter_uniforms.append(None)
            else:
                self.parameter_uniforms.append(glGetUniformLocation(self.program, 'params' + str(i)))

        glUseProgram(self.program)
        for i, (name, filename) in enumerate(images.items()):
            glActiveTexture(GL_TEXTURE0 + (2 + i))
            texture = load_image(filename)
            glUniform1i(glGetUniformLocation(self.program, name), 2 + i)
            self.textures[name] = texture

    def reset_all(self):
        self.reset()

        min_bound, max_bound = self.get_bounds()

        positions = np.random.random((self.particles, 2)).astype(np.float32)
        positions *= max_bound - min_bound
        positions += min_bound

        if self.uses_markov_chain:
            particle_data = (ParticleWithMarkov * self.particles)()
        else:
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

            if self.uses_markov_chain and len(self.functions) != 0:
                entry.prev = i % len(self.functions)

        for buffer in self.buffers:
            glBindBuffer(GL_ARRAY_BUFFER, buffer)
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(particle_data), particle_data)

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
        func = Function(function_formats)
        self.functions[func] = 0.0
        return func

    def set_use_markov_chain(self, enabled):
        if enabled == self.uses_markov_chain:
            return

        self.uses_markov_chain = enabled

        for i, func in enumerate(self.functions):
            if enabled:
                vector = [0] * len(self.functions)
                vector[i] = 1
                self.functions[func] = vector
            else:
                prob = 0
                if i == 0:
                    prob = 1
                self.functions[func] = prob

        self.create_buffers()
        self.generate_program()

        self.reset_all()

    def create_buffers(self):
        self.buffers = glGenBuffers(2).tolist()

        if self.uses_markov_chain:
            buf_size = sizeof(ParticleWithMarkov) * self.particles
        else:
            buf_size = sizeof(Particle) * self.particles

        for buffer in self.buffers:
            glBindBuffer(GL_ARRAY_BUFFER, buffer)
            glBufferData(GL_ARRAY_BUFFER, buf_size, None, GL_STATIC_DRAW)
