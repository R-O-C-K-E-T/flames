import ctypes
import re
from os import path

from OpenGL.GL import (
    GL_FRAGMENT_SHADER,
    GL_TEXTURE0,
    GL_TEXTURE1,
    GL_TEXTURE_2D,
    GL_TEXTURE_2D_ARRAY,
    GL_TEXTURE_DEPTH,
    GL_TEXTURE_HEIGHT,
    GL_TEXTURE_WIDTH,
    GL_TRIANGLE_STRIP,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glBindTexture,
    glDrawArrays,
    glGetTexLevelParameteriv,
    glGetUniformfv,
    glGetUniformLocation,
    glUniform1f,
    glUniform1i,
    glUniform1ui,
    glUniform4fv,
    glUseProgram,
    shaders,
)


def compile_shader(text, shader_type, defines={}):
    def add_globals(string):
        header = ''
        for key, val in defines.items():
            if isinstance(val, bool):
                if val:
                    header += '#define {}\n'.format(key)
                else:
                    header += '#undef {}\n'.format(key)
            else:
                header += '#define {} {}\n'.format(key, val)

        index = string.index('\n') + 1
        return string[:index] + header + string[index:]

    text = add_globals(text)
    # return shaders.compileShader(text, type)
    try:
        return shaders.compileShader(text, shader_type)
    except RuntimeError as e:
        lines = text.split('\n')
        for cause in e.args[0].split('\\n'):
            print(cause)
            match = re.search('0\\(([0-9]+)\\)', cause)
            if match is None:
                continue
            line = int(match[1]) - 1
            print(*lines[line - 1 : line + 2], sep='\n')
    raise RuntimeError('Compilation Failed')


def get_texture_size_2d(texture):
    glBindTexture(GL_TEXTURE_2D, texture)
    width = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
    height = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)
    glBindTexture(GL_TEXTURE_2D, 0)

    return width, height


def get_texture_size_2d_array(texture):
    glBindTexture(GL_TEXTURE_2D_ARRAY, texture)
    width = glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_WIDTH)
    height = glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_HEIGHT)
    depth = glGetTexLevelParameteriv(GL_TEXTURE_2D_ARRAY, 0, GL_TEXTURE_DEPTH)
    glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    return width, height, depth


class FlameRenderer:
    def __init__(self, colour_texture, histogram_texture, background_colour=(0, 0, 0, 1)):
        self.colour = colour_texture
        self.histogram = histogram_texture

        width, height, self.supersamples = get_texture_size_2d_array(self.colour)
        self.size = width, height

        unrolled_supersampling = ''.join(
            'result += texture(colour, vec3(texCoord, {})).rgb;'.format(i) for i in range(self.supersamples)
        )
        defines = {
            'SUPERSAMPLES': self.supersamples,
            'UNROLLED_SUPERSAMPLING': unrolled_supersampling,
        }

        current_dir = path.dirname(__file__)
        vert = compile_shader(open(path.join(current_dir, 'flames.vert')).read(), GL_VERTEX_SHADER)
        frag = compile_shader(open(path.join(current_dir, 'flames.frag')).read(), GL_FRAGMENT_SHADER, defines=defines)

        self.program = shaders.compileProgram(vert, frag, validate=False)

        glUseProgram(self.program)
        glUniform1i(glGetUniformLocation(self.program, 'histogram'), 0)
        glUniform1i(glGetUniformLocation(self.program, 'colour'), 1)

        # self.tick_uniform = glGetUniformLocation(self.program, 'tick')
        self.maximum_val_uniform = glGetUniformLocation(self.program, 'maximumVal')
        self.gamma_uniform = glGetUniformLocation(self.program, 'gamma')
        self.background_uniform = glGetUniformLocation(self.program, 'background')

        self.program.check_validate()

        self.set_background_colour(background_colour)

    def set_background_colour(self, val):
        glUseProgram(self.program)
        glUniform4fv(self.background_uniform, 1, val)

    def get_background_colour(self):
        arr = (ctypes.c_float * 4)()
        glGetUniformfv(self.program, self.background_uniform, arr)
        return list(arr)

    def render(self, maximum_val, gamma):
        glUseProgram(self.program)

        # glUniform1ui(self.tick_uniform, tick)
        glUniform1ui(self.maximum_val_uniform, maximum_val)
        glUniform1f(self.gamma_uniform, gamma)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.histogram)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.colour)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
