import argparse
import copy
import glob
import json
import math
import os
import os.path
import shutil
import subprocess
import sys
import time

import numpy as np


def de_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # assume no object arrays

    if isinstance(obj, list):
        return [de_numpy(item) for item in obj]

    if isinstance(obj, dict):
        return dict((de_numpy(key), de_numpy(val)) for key, val in obj.items())

    return obj


def left_pad(digits, n):
    # TODO Use 3rd party library for this
    v = str(n)
    while len(v) < digits:
        v = '0' + v
    return v


def rot_mat(angle):
    s = math.sin(angle)
    c = math.cos(angle)
    return np.array([[c, -s], [s, c]])


def main():
    parser = argparse.ArgumentParser(description='Render a 3d fractal flame animation')

    parser.add_argument('flame', type=str, help='Base flame to animate')
    parser.add_argument('animation', type=str, help='Python script to permute the flame')

    parser.add_argument(
        '--iterations',
        type=int,
        default=100_000_000,
        help='Number of iterations to perform for each frame',
    )
    parser.add_argument('--frames', type=int, default=300, help='Number of frames to render')
    parser.add_argument('--framerate', type=int, default=30, help='Frames per second of video')
    parser.add_argument('--width', type=int, default=500, help='Width of output video')
    parser.add_argument('--height', type=int, default=500, help='Height of output video')
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Skip rendering flames and only produce JSON flame descriptions for each flame',
    )

    args = parser.parse_args()

    initial = time.time()

    flame_directory = 'tmp-flame'
    render_directory = 'tmp-render'
    out_filename = os.path.splitext(os.path.basename(args.animation))[0] + '.mp4'

    try:
        shutil.rmtree(flame_directory)
    except FileNotFoundError:
        pass
    try:
        shutil.rmtree(render_directory)
    except FileNotFoundError:
        pass

    os.makedirs(flame_directory)
    os.makedirs(render_directory)

    iterations = args.iterations
    width = args.width
    height = args.height
    framerate = args.framerate

    frames = args.frames
    duration = frames / framerate
    digits = math.ceil(math.log(frames, 10))

    with open(args.flame, 'r') as flame_file:
        original_flame = json.load(flame_file)
    with open(args.animation, 'r') as animation_file:
        code = compile(animation_file.read(), args.animation, 'exec')

    for i in range(frames):
        flame = copy.deepcopy(original_flame)
        t = i / frames

        for func in flame['functions']:
            func['pre_trans'] = np.array(func['pre_trans'])
            func['post_trans'] = np.array(func['post_trans'])

        exec(
            code,
            {
                'math': math,
                't': t,
                'flame': flame,
                'rot_mat': rot_mat,
            },
            {},
        )

        for func in flame['functions']:
            func['pre_trans'] = func['pre_trans'].tolist()
            func['post_trans'] = func['post_trans'].tolist()

        with open(os.path.join(flame_directory, left_pad(digits, i) + '.json'), 'w') as f:
            json.dump(flame, f)

    if args.no_render:
        exit()

    subprocess.run(
        [
            sys.executable,
            '-m',
            'flames.render',
            '--destination',
            render_directory,
            '--iterations',
            str(iterations),
            '--width',
            str(width),
            '--height',
            str(height),
            *glob.glob(f'{flame_directory}/*.json'),
        ],
        check=True,
    )
    try:
        os.remove(out_filename)
    except FileNotFoundError:
        pass
    subprocess.run(
        [
            'ffmpeg',
            '-framerate',
            str(framerate),
            '-pattern_type',
            'glob',
            '-i',
            f'{render_directory}/*.png',
            '-c:v',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            out_filename,
        ],
        check=True,
    )

    dt = time.time() - initial
    print('Animation saved to {} in {:.3f}s {:.3f}x'.format(out_filename, dt, duration / dt))


if __name__ == '__main__':
    main()
