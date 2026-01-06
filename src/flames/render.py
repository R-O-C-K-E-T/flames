#!/usr/bin/env python3

import argparse
import json
import os
import time
import traceback
from contextlib import contextmanager
from os import path

import pygame
from OpenGL.GL import (
    GL_ALL_BARRIER_BITS,
    GL_COLOR_ATTACHMENT0,
    GL_FRAMEBUFFER,
    GL_NEAREST,
    GL_RGB,
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    GL_VIEWPORT,
    glBindFramebuffer,
    glBindTexture,
    glDeleteFramebuffers,
    glDeleteTextures,
    glFinish,
    glFramebufferTexture,
    glGenFramebuffers,
    glGenTextures,
    glGetIntegerv,
    glGetTexImage,
    glMemoryBarrier,
    glTexImage2D,
    glTexParameteri,
    glViewport,
)
from PIL import Image
from pygame.locals import OPENGL

from flames.backend import create_flame_iterator
from flames.backend.common import FlameIterator, FunctionFormat
from flames.renderer import FlameRenderer


def load_file(flame_str: str, iterator: FlameIterator, base_directory=None):
    content = json.loads(flame_str)

    iterator.set_use_markov_chain(isinstance(content['functions'][0]['prob'], list))

    iterator.functions = {}

    iterator.centre[:] = content['pos']
    iterator.radius = content['radius']

    for func_data in content['functions']:
        params = []
        func_formats = []
        for var_data in func_data['variations']:
            params += list(var_data['params'].values())

            images = var_data.get('images', {})
            if base_directory is not None:
                for name in images:
                    images[name] = path.join(base_directory, images[name])

            func_formats.append(
                FunctionFormat(
                    var_data['name'],
                    var_data['base'],
                    list(var_data.get('params', {}).keys()),
                    images,
                )
            )

        func = iterator.create_function(func_formats)

        prob = func_data['prob']
        if isinstance(prob, list) != iterator.uses_markov_chain:
            raise ValueError('Incompatible iterator')

        if isinstance(prob, list) and len(prob) != len(content['functions']):
            raise ValueError('Size mismatch between markov chain transition length')

        iterator.functions[func] = prob

        func.params[:] = params
        colour = func_data['colour']
        if len(colour) < 4:
            colour.append(0.5)
        func.colour[:] = colour
        func.pre_trans[:] = func_data['pre_trans']
        func.post_trans[:] = func_data['post_trans']
    iterator.reset_all()

    return content['gamma'], content['hits_per_tick']


@contextmanager
def render_to_file(file, size, alpha):
    prev_viewport = glGetIntegerv(GL_VIEWPORT)

    framebuffer = int(glGenFramebuffers(1))
    texture = int(glGenTextures(1))

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)

    glBindTexture(GL_TEXTURE_2D, texture)
    if alpha:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, *size, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    else:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, *size, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texture, 0)

    glViewport(0, 0, *size)
    try:
        yield None
    finally:
        glMemoryBarrier(GL_ALL_BARRIER_BITS)  # Dunno man

        glBindTexture(GL_TEXTURE_2D, texture)
        if alpha:
            pixel_data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE)
            img = Image.frombytes('RGBA', size, pixel_data)
        else:
            pixel_data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE)
            img = Image.frombytes('RGB', size, pixel_data)

        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        if isinstance(file, str):
            with open(file, 'wb') as f:
                img.save(f)
        else:
            img.save(file)

        glDeleteFramebuffers(1, [framebuffer])
        glDeleteTextures([texture])

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(*prev_viewport)


def main():
    parser = argparse.ArgumentParser(description='Render a flame file')

    parser.add_argument('flames', type=str, nargs='+', help='Flame(s) to render')

    parser.add_argument('--destination', type=str, default='.', help='Output image directory')
    parser.add_argument('--width', type=int, default=1000, help='Output image width')
    parser.add_argument('--height', type=int, default=1000, help='Output image height')
    parser.add_argument('--supersamples', type=int, default=4, help='Output image supersamples')
    parser.add_argument('--alpha', action='store_true', help='Flag to enable output image alpha')
    parser.add_argument('--iterations', type=int, default=10_000_000_000, help='Number of iterations to perform')
    parser.add_argument('--backend', type=str, default='basic', help='Flame Iterator to use')

    args = parser.parse_args()

    if args.width <= 0:
        raise RuntimeError('width <= 0')
    if args.height <= 0:
        raise RuntimeError('height <= 0')
    if args.supersamples <= 0:
        raise RuntimeError('supersamples <= 0')
    if args.iterations <= 0:
        raise RuntimeError('iterations <= 0')

    completed = 0

    initial = time.time()
    pygame.display.set_mode((1, 1), OPENGL)
    try:
        iterator = create_flame_iterator(args.backend, (args.width, args.height), args.supersamples)
        renderer = FlameRenderer(iterator.colour, iterator.histogram, (0, 0, 0, 0) if args.alpha else (0, 0, 0, 1))

        for in_file in args.flames:
            t = time.time()

            try:
                with open(in_file, 'r') as file:
                    flame_str = file.read()
            except OSError:
                print('Failed to load:', in_file)
                traceback.print_exc()
                continue

            try:
                gamma, hits_per_tick = load_file(flame_str, iterator, path.dirname(in_file))
            except ValueError:
                print('Failed to interpret:', in_file)
                traceback.print_exc()
                continue

            dest_name = path.join(args.destination, path.splitext(path.basename(in_file))[0] + '.png')
            try:
                os.makedirs(path.dirname(dest_name))
            except FileExistsError:
                pass

            max_iterations = args.iterations // iterator.particles
            block_size = 1000
            for _ in range(max_iterations // block_size):
                for _ in range(block_size):
                    iterator.update()
                glFinish()
            for _ in range(max_iterations % block_size):
                iterator.update()

            maximum_val = hits_per_tick * iterator.tick * iterator.particles / (iterator.size[0] * iterator.size[1])

            try:
                dest_file = open(dest_name, 'wb')
            except OSError:
                print('Failed to open:', dest_name)
                continue

            with dest_file, render_to_file(dest_file, (args.width, args.height), args.alpha):
                renderer.render(int(maximum_val), gamma)

            dt = time.time() - t
            completed += 1
            print(
                'Rendered {} to {} in {:.2f}s @ {:,} iterations per second'.format(
                    in_file, dest_name, dt, round(args.iterations / dt)
                ),
            )
        print('{}/{} rendered in {:.2f}s'.format(completed, len(args.flames), time.time() - initial))

        if len(args.flames) > 0 and completed == 0:
            exit(1)  # Everything failed
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
