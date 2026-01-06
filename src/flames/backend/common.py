from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from OpenGL.GL import (
    GL_CLAMP_TO_EDGE,
    GL_NEAREST,
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_TEXTURE_2D_ARRAY,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_UNSIGNED_BYTE,
    glBindTexture,
    glGenTextures,
    glTexImage2D,
    glTexParameteri,
)
from PIL import Image


class FunctionFormat:
    def __init__(self, name: str, base: str, params: list[str] = [], images: dict[str, str] = {}):
        self.name = name
        self.base = base
        self.params = params.copy()  # Variable names, All floats
        self.images = images.copy()  # name -> filename

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.base == other.base
            and self.params == other.params
            and self.images == other.images
        )

    def __hash__(self):
        return hash((self.base, tuple(self.params), tuple(self.images)))

    def instantiate(self, input_names, name_generator, image_names):
        assert len(self.params) == len(input_names)

        result = self.base
        for param, input_name in zip(self.params, input_names):
            result = result.replace(param, input_name)
        for og_image, image in zip(self.images, image_names):
            result = result.replace(og_image, image)

        return '{' + result + '}'


def load_image(filename):
    img = Image.open(filename)
    img = img.transpose(Image.FLIP_TOP_BOTTOM).convert('RGBA')
    data = img.tobytes()
    texture = gen_texture_2d()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, *img.size, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    return texture


def gen_texture_2d():
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    return texture


def gen_texture_2d_array():
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D_ARRAY, texture)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    return texture


HELPERS = """// RANDOM = uniform random float in [0,1)
// RAND_VEC = vec2(RANDOM, RANDOM)
float radius2 = dot(coord, coord);
float radius = sqrt(radius2);
float theta = atan(coord.x, coord.y);
float phi = (PI/2) - theta;"""

BASE_FUNCTIONS = [
    FunctionFormat('Linear', 'RETURN(coord * weight)', ['weight']),
    FunctionFormat('Sinusoidal', 'RETURN(vec2(sin(coord.x), sin(coord.y)) * weight)', ['weight']),
    FunctionFormat('Spherical', 'RETURN(coord / radius2 * weight)', ['weight']),
    FunctionFormat(
        'Swirl',
        """RETURN(vec2(
            coord.x*sin(radius2) - coord.y*cos(radius2),
            coord.x*cos(radius2) + coord.y*sin(radius2)
        ) * weight)""",
        ['weight'],
    ),
    FunctionFormat(
        'Horseshoe',
        'RETURN(vec2((coord.x-coord.y) * (coord.x+coord.y), 2*coord.x*coord.y) / radius * weight)',
        ['weight'],
    ),
    FunctionFormat('Polar', 'RETURN(vec2(theta / PI, radius - 1) * weight)', ['weight']),
    FunctionFormat(
        'Handkerchief',
        'RETURN(radius * vec2(sin(theta + radius), cos(theta - radius)) * weight)',
        ['weight'],
    ),
    FunctionFormat('Heart', 'RETURN(radius*vec2(sin(theta*radius), -cos(theta*radius)) * weight)', ['weight']),
    FunctionFormat('Disc', 'RETURN(theta / PI * vec2(sin(PI*radius), cos(PI*radius)) * weight)', ['weight']),
    FunctionFormat(
        'Spiral',
        'RETURN(vec2(cos(theta) + sin(radius), sin(theta) - cos(radius)) / radius * weight)',
        ['weight'],
    ),
    FunctionFormat('Hyperbolic', 'RETURN(vec2(sin(theta) / radius, cos(theta) * radius) * weight)', ['weight']),
    FunctionFormat(
        'Diamond',
        'RETURN(vec2(sin(theta) * cos(radius), cos(theta) * sin(radius)) * weight)',
        ['weight'],
    ),
    FunctionFormat(
        'Ex',
        """
float p0 = sin(theta + radius);
float p1 = sin(theta - radius);
RETURN((pow(p0,3) + vec2(pow(p1,3), -pow(p1,3))) * weight)
        """,
        ['weight'],
    ),
    FunctionFormat(
        'Julia',
        """
float omega;
if (RANDOM < 0.5) {
    omega=theta/2;
} else {
    omega = theta/2 + PI;
}
RETURN(sqrt(radius) * vec2(cos(omega), sin(omega)) * weight)
        """,
        ['weight'],
    ),
    FunctionFormat(
        'Bent',
        'RETURN(vec2((coord.x >= 0) ? coord.x : coord.x*2, (coord.y >= 0) ? coord.y : coord.y*0.5) * weight)',
        ['weight'],
    ),
    FunctionFormat(
        'Waves',
        """RETURN(vec2(
            coord.x + amplitudeX*sin(coord.y / (wavelengthX*wavelengthX)),
            coord.y + amplitudeY*sin(coord.x / (wavelengthY*wavelengthY))
        ) * weight)""",
        ['weight', 'amplitudeX', 'wavelengthX', 'amplitudeY', 'wavelengthY'],
    ),
    FunctionFormat('Fisheye', 'RETURN(vec2(coord.y, coord.x) * (2*weight / (radius + 1)))', ['weight']),
    FunctionFormat(
        'Popcorn',
        'RETURN(vec2(coord.x + amplitudeX*sin(tan(3*coord.y)), coord.y + amplitudeY*sin(tan(3*coord.x))) * weight)',
        ['weight', 'amplitudeX', 'amplitudeY'],
    ),
    FunctionFormat(
        'Exponential',
        'RETURN(vec2(cos(PI*coord.y), sin(PI*coord.y)) * (weight * exp(coord.x - 1)))',
        ['weight'],
    ),
    FunctionFormat(
        'Power',
        'RETURN(vec2(cos(theta), sin(theta)) * (pow(radius, sin(theta)) * weight))',
        ['weight'],
    ),
    FunctionFormat(
        'Cosine',
        'RETURN(vec2(cos(PI*coord.x)*cosh(coord.y), -sin(PI*coord.x)*sinh(coord.y)) * weight)',
        ['weight'],
    ),
    FunctionFormat(
        'Rings',
        """
float str2 = strength*strength;
RETURN(vec2(cos(theta), sin(theta)) * ((mod(radius + str2, 2*str2) - str2 + radius*(1-str2)) * weight))
""",
        ['weight', 'strength'],
    ),
    FunctionFormat(
        'Fan',
        """float paramT = PI * strength * strength;
if (mod(theta + offset, paramT) > paramT * 0.5) {
    RETURN(vec2(cos(theta - paramT*0.5), sin(theta - paramT*0.5)) * (radius * weight))
} else {
    RETURN(vec2(cos(theta + paramT*0.5), sin(theta + paramT*0.5)) * (radius * weight))
}""",
        ['weight', 'strength', 'offset'],
    ),
    FunctionFormat(
        'Blob',
        'RETURN(vec2(cos(theta), sin(theta)) * ((low + (high - low)*(sin(count*theta) + 1)*0.5) * radius * weight))',
        ['weight', 'high', 'low', 'count'],
    ),
    FunctionFormat(
        'PDJ',
        'RETURN(vec2(sin(paramA*coord.y) - cos(paramB*coord.x), sin(paramC*coord.x) - cos(paramD*coord.y)) * weight)',
        ['weight', 'paramA', 'paramB', 'paramC', 'paramD'],
    ),
    FunctionFormat(
        'Fan2',
        """float pA2 = PI*paramA*paramA;
float paramT = theta + paramB - pA2*fract(2*theta*pA2/paramB);
if (paramT > pA2*0.5) {
    RETURN(vec2(sin(theta - pA2*0.5), cos(theta - pA2*0.5)) * (radius * weight))
} else {
    RETURN(vec2(sin(theta + pA2*0.5), cos(theta + pA2*0.5)) * (radius * weight))
}""",
        ['weight', 'paramA', 'paramB'],
    ),
    FunctionFormat(
        'Rings2',
        """float str2 = strength * strength;
float paramT = radius - 2*str2*fract((radius + str2) / (2*str2)) + radius*(1 - str2);
RETURN(vec2(sin(theta), cos(theta)) * (paramT * weight))
""",
        ['weight', 'strength'],
    ),
    FunctionFormat('EyeFish', 'RETURN(coord * (2*weight / (radius + 1)))', ['weight']),
    FunctionFormat('Bubble', 'RETURN(coord * (4*weight / (radius2 + 4)))', ['weight']),
    FunctionFormat('Cylinder', 'RETURN(vec2(sin(coord.x), coord.y) * weight)', ['weight']),
    FunctionFormat(
        'Perspective',
        'RETURN(vec2(coord.x, coord.y*cos(angle)) * (weight*dist / (dist - coord.y*sin(angle))))',
        ['weight', 'angle', 'dist'],
    ),
    FunctionFormat(
        'Noise',
        'float angle = 2*PI*RANDOM;\nRETURN(vec2(coord.x*cos(angle), coord.y*sin(angle)) * (weight * RANDOM))',
        ['weight'],
    ),
    FunctionFormat(
        'JuliaN',
        """float p3 = fract(abs(power)*RANDOM);
float p4 = (phi + 2*PI*p3) / power;
RETURN(vec2(cos(p4), sin(p4)) * (weight * pow(radius, dist / power)))""",
        ['weight', 'power', 'dist'],
    ),
    FunctionFormat(
        'JuliaScope',
        """float delta = (RANDOM < 0.5) ? -1 : 1;
float p3 = fract(abs(power)*RANDOM);
float p4 = (delta*phi + 2*PI*p3) / power;
RETURN(vec2(cos(p4), sin(p4)) * (weight * pow(radius, dist/power)))""",
        ['weight', 'power', 'dist'],
    ),
    FunctionFormat(
        'Blur',
        'float angle = 2*PI*RANDOM;\nRETURN(vec2(cos(angle), sin(angle)) * (weight*RANDOM))',
        ['weight'],
    ),
    FunctionFormat(
        'Guassian',
        """
float angle = 2*PI*RANDOM;
RETURN(vec2(cos(angle),sin(angle)) * (weight * (RANDOM + RANDOM + RANDOM + RANDOM - 2)))
        """,
        ['weight'],
    ),
    FunctionFormat(
        'RadialBlur',
        """float p0 = weight * (RANDOM + RANDOM + RANDOM + RANDOM - 2);
float p1 = phi + p0*sin(angle);
float p2 = p0*cos(angle) - 1;
RETURN(vec2(radius*cos(p1) + p2*coord.x, radius*sin(p1) + p2*coord.y))""",
        ['weight', 'angle'],
    ),
    FunctionFormat(
        'Pie',
        """float p0 = fract(RANDOM*slices + 0.5);
float p1 = rotation + (p0 + RANDOM*thickness) * 2*PI / slices;
RETURN(vec2(cos(p1), sin(p1)) * (RANDOM*weight))""",
        ['weight', 'slices', 'rotation', 'thickness'],
    ),
    FunctionFormat(
        'Ngon',
        """float angle = 2*PI / sides;
float p0 = phi - angle*floor(phi / sides);
float p1 = p0;
if (p0 <= sides*0.5) {
    p1 -= angle;
}
RETURN(coord * (weight * (corners*(1/cos(p1) - 1) + circle) * pow(radius, -power)))
        """,
        ['weight', 'power', 'sides', 'corners', 'circle'],
    ),
    FunctionFormat(
        'Curl',
        """
float p0 = 1 + paramA*coord.x + paramB*(coord.x*coord.x - coord.y*coord.y);
float p1 = paramA*coord.y + 2*paramB*coord.x*coord.y;
RETURN(vec2(coord.x*p0 + coord.y*p1, coord.y*p0 - coord.x*p1) * (weight / (p0*p0 + p1*p1)))
        """,
        ['weight', 'paramA', 'paramB'],
    ),
    FunctionFormat(
        'Rectangles',
        """
RETURN(vec2(
    (2*floor(coord.x/sizeX) + 1)*sizeX - coord.x,
    (2*floor(coord.y/sizeY) + 1)*sizeY - coord.y
) * weight)
""",
        ['weight', 'sizeX', 'sizeY'],
    ),
    FunctionFormat(
        'Arch',
        'float angle = RANDOM*PI*weight;\nRETURN(vec2(sin(angle), sin(angle)*sin(angle)/cos(angle)) * weight)',
        ['weight'],
    ),
    FunctionFormat('Tangent', 'RETURN(vec2(sin(coord.x) / cos(coord.y),tan(coord.y)) * weight)', ['weight']),
    FunctionFormat('Square', 'RETURN(vec2(RANDOM - 0.5, RANDOM - 0.5) * weight)', ['weight']),
    FunctionFormat(
        'Rays',
        'RETURN(vec2(cos(coord.x), sin(coord.y)) * (weight*weight*tan(RANDOM*PI*weight) / radius2))',
        ['weight'],
    ),
    FunctionFormat(
        'Blade',
        """
float angle = RANDOM*radius*weight;
RETURN(vec2(cos(angle) + sin(angle), cos(angle) - sin(angle)) * (coord.x*weight))
        """,
        ['weight'],
    ),
    FunctionFormat('Secant', 'RETURN(vec2(coord.x, 1.0/(weight*cos(weight*radius))) * weight)', ['weight']),
    FunctionFormat(
        'Twintrian',
        """
float angle = RANDOM*radius*weight;
float val = log(sin(angle))*(2.0/log(10.0)) + cos(angle);
RETURN(vec2(val, val-PI*sin(angle)) * (coord.x*weight))
        """,
        ['weight'],
    ),
    FunctionFormat('Cross', 'RETURN(coord * (weight / abs(coord.x*coord.x - coord.y*coord.y)))', ['weight']),
]


class FlameIterator(ABC):
    def __init__(self, size: tuple[int, int], supersamples: int):
        self.size = np.array(size)
        self.supersamples = supersamples
        self.functions = {}
        self.radius = 0.0
        self.centre = np.zeros(2)
        super().__init__()

    @abstractmethod
    def update(self) -> None: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def reset_all(self) -> None: ...

    @abstractmethod
    def set_use_markov_chain(self, enabled: bool) -> None: ...

    @abstractmethod
    def create_function(self, function_formats: list[FunctionFormat]) -> Any: ...
