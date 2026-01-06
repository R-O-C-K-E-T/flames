#!/usr/bin/env python3

import argparse
import copy
import json
import os
from os import path

import numpy as np


def interpolation_functions(func_a, func_b):
    variations = []
    for func in (func_a, func_b):
        variations.append(
            {
                (var['name'], var['base'], frozenset(var['params'])): var
                for var in func['variations']
            }
        )
    variations_a, variations_b = variations

    keys_a = set(variations_a)
    keys_b = set(variations_b)

    exclusive_a = [variations_a[key] for key in (keys_a - keys_b)]
    shared = [(variations_a[key], variations_b[key]) for key in (keys_a & keys_b)]
    exclusive_b = [variations_b[key] for key in (keys_b - keys_a)]

    for var in exclusive_a + exclusive_b:
        if 'weight' not in var['params']:
            raise RuntimeError(
                'All variations not in common between two similar functions must have a weight parameter'
            )

    def interpolate(u):
        variations = []

        for var in exclusive_a:
            var = copy.deepcopy(var)
            var['params']['weight'] *= 1 - u
            variations.append(var)

        for var_a, var_b in shared:
            var = copy.copy(var_a)
            params = {}
            for name in var_a['params']:
                params[name] = var_a['params'][name] * (1 - u) + var_b['params'][name] * u
            var['params'] = params
            variations.append(var)

        for var in exclusive_b:
            var = copy.deepcopy(var)
            var['params']['weight'] *= u
            variations.append(var)
        return {
            'name': func_a['name'],
            'prob': func_a['prob'] * (1 - u) + func_b['prob'] * u,
            'colour': (np.array(func_a['colour']) * (1 - u) + np.array(func_b['colour']) * u).tolist(),
            'pre_trans': (np.array(func_a['pre_trans']) * (1 - u) + np.array(func_b['pre_trans']) * u).tolist(),
            'post_trans': (np.array(func_a['post_trans']) * (1 - u) + np.array(func_b['post_trans']) * u).tolist(),
            'variations': variations,
        }

    return interpolate


def main():
    parser = argparse.ArgumentParser(description='Interpolate between two flame files')

    parser.add_argument('start_flame', type=str, help='Starting flame')
    parser.add_argument('end_flame', type=str, help='Ending flame')

    parser.add_argument('--frames', type=int, default=100, help='Number of frames')
    parser.add_argument('--output', type=str, default='tmp-flame', help='Output directory')
    parser.add_argument('--include-end', action='store_true')

    args = parser.parse_args()

    if args.include_end:
        if args.frames <= 1:
            raise RuntimeError('Frame count should be greater than 1')
    else:
        if args.frames <= 0:
            raise RuntimeError('Frame count should be greater than 0')

    os.makedirs(args.output, exist_ok=True)

    with open(args.start_flame, 'r') as flame_file:
        flame_a = json.load(flame_file)
    with open(args.end_flame, 'r') as flame_file:
        flame_b = json.load(flame_file)

    functions_a = dict((func['name'], func) for func in flame_a['functions'])
    functions_b = dict((func['name'], func) for func in flame_b['functions'])

    if len(functions_a) != len(flame_a['functions']):
        raise RuntimeError('Start flame has repeated function name')
    if len(functions_b) != len(flame_b['functions']):
        raise RuntimeError('End flame has repeated function name')

    names_a = set(functions_a)
    names_b = set(functions_b)

    exclusive_a = [functions_a[name] for name in (names_a - names_b)]
    shared = [interpolation_functions(functions_a[name], functions_b[name]) for name in (names_a & names_b)]
    exclusive_b = [functions_b[name] for name in (names_b - names_a)]

    divisor = args.frames
    if args.include_end:
        divisor -= 1

    for i in range(args.frames):
        u = i / divisor

        functions = []
        for func in exclusive_a:
            func = func.copy()
            func['prob'] *= 1 - u
            functions.append(func)

        functions += [iterpolator(u) for iterpolator in shared]

        for func in exclusive_b:
            func = func.copy()
            func['prob'] *= u
            functions.append(func)

        out_flame = {
            'pos': (np.array(flame_a['pos']) * (1 - u) + np.array(flame_b['pos']) * u).tolist(),
            'radius': flame_a['radius'] * (1 - u) + flame_b['radius'] * u,
            'gamma': flame_a['gamma'] * (1 - u) + flame_b['gamma'] * u,
            'hits_per_tick': flame_a['hits_per_tick'] * (1 - u) + flame_b['hits_per_tick'] * u,
            'functions': functions,
        }

        with open(path.join(args.output, str(i) + '.json'), 'w') as f:
            json.dump(out_flame, f)


if __name__ == '__main__':
    main()
