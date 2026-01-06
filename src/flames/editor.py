#!/usr/bin/env python3

import argparse
import copy
import json
import math
import multiprocessing as mp
import operator
import queue
import random
import re
import threading
import time
import tkinter as tk
from functools import partial, reduce
from os import path
from tkinter import filedialog, messagebox, ttk
from traceback import print_exc

import numpy as np

from flames.backend import BackendMethod, create_flame_iterator
from flames.backend.common import BASE_FUNCTIONS, HELPERS, FunctionFormat

IDENTIY_MAT = [1, 0, 0, 1, 0, 0]
event_queue = mp.Queue()


def raise_window(window):
    window.attributes('-topmost', 1)
    window.attributes('-topmost', 0)


def length(vec):
    return math.sqrt(vec[0] ** 2 + vec[1] ** 2)


def rotate(vec, angle):
    sin = math.sin(angle)
    cos = math.cos(angle)
    return np.array([vec[0] * cos - vec[1] * sin, vec[0] * sin + vec[1] * cos], dtype=float)


def hsv_to_rgb(hue, sat, val):
    c = val * sat
    hue = (hue % 1) * 6
    x = c * (1 - abs((hue % 2) - 1))
    if hue > 5:
        out = c, 0, x
    elif hue > 4:
        out = x, 0, c
    elif hue > 3:
        out = 0, x, c
    elif hue > 2:
        out = 0, c, x
    elif hue > 1:
        out = x, c, 0
    else:
        out = c, x, 0
    return np.array(out) + val - c


def rgb_to_hsv(r, g, b):
    delta = max(r, g, b) - min(r, g, b)
    if r == g == b:
        hue = 0
    else:
        if r > g and r > b:
            hue = (0 + (g - b) / delta) / 6
        elif g > b:
            hue = (2 + (b - r) / delta) / 6
        else:
            hue = (4 + (r - g) / delta) / 6
    hue = hue % 1

    if r == g == b == 0:
        sat = 0
    else:
        sat = delta / max(r, g, b)
    val = max(r, g, b)
    return np.array([hue, sat, val], float)


def convert_colour(colour):
    return '#%02x%02x%02x' % tuple(min(math.floor(256 * i), 255) for i in colour)


def rotation(angle):
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


def approximate_markov_steady_state(transitions):
    power = np.linalg.matrix_power(transitions, 100)

    approx_prob = power @ ([1] * len(transitions))
    approx_prob /= np.sum(approx_prob)

    return approx_prob


def fix_probabilities(probs):
    total = sum(probs)
    if total == 0:
        return [1 / len(probs)] * len(probs)
    else:
        return [p / total for p in probs]


class ColourPicker(tk.Toplevel):
    def __init__(self, initial_colour=(1, 1, 1, 0.5), title='Colour Picker'):
        super().__init__()
        self.size = 150

        self.resizable(False, False)
        self.title(title)
        # self.attributes('-topmost', True)

        self.protocol('WM_DELETE_WINDOW', self.cancel)

        self.canvas = tk.Canvas(self, width=self.size + 20, height=self.size + 12, bg='white')
        self.canvas.bind('<B1-Motion>', self.move)
        self.canvas.bind('<Button-1>', self.click)
        self.canvas.bind('<ButtonRelease-1>', self.release)

        self.img = tk.PhotoImage(width=self.size, height=self.size)
        self.canvas.create_image((0, 0), anchor=tk.NW, image=self.img, state='normal')

        self.base_wheel = np.array(
            [
                [
                    hsv_to_rgb(math.atan2(y, x) / math.pi / 2, math.sqrt(x**2 + y**2) / self.size * 2, 1)
                    if x**2 + y**2 <= self.size * self.size / 4
                    else (-1, -1, -1)
                    for x in range(-self.size // 2, self.size // 2)
                ]
                for y in range(-self.size // 2, self.size // 2)
            ]
        )
        self.format_str = ' '.join(
            '{' + ' '.join('#%02x%02x%02x' for _ in range(self.size)) + '}' for _ in range(self.size)
        )

        self.canvas.create_line(
            self.size + 10,
            10,
            self.size + 10,
            self.size - 10,
            width=5,
            capstyle=tk.ROUND,
            fill=convert_colour((0.4, 0.4, 0.4)),
        )

        self.canvas.grid(row=0, column=0, columnspan=2)

        self.cursors = [None] * 2

        self.initial_alpha = self.alpha = initial_colour[3]

        self.rgb_controls = [tk.StringVar(self) for i in range(3)]
        self.hsv_controls = [tk.StringVar(self) for i in range(3)]
        self.alpha_control = tk.StringVar(self, value=str(initial_colour[3]))
        tk.Spinbox(
            self,
            from_=0,
            to=1,
            increment=0.1,
            width=8,
            textvariable=self.alpha_control,
        ).grid(row=4, column=0, columnspan=2)

        def update_alpha(*_):
            try:
                self.alpha = float(self.alpha_control.get())
                self.update_colour()
            except ValueError:
                pass

        self.alpha_control.trace('w', update_alpha)

        def update_hsv(*_):
            if self.selected is not None:
                return
            try:
                new_column = [float(channel.get()) for channel in self.rgb_controls]
            except ValueError:
                return
            if min(new_column) < 0 or max(new_column) > 1:
                return

            new_column = rgb_to_hsv(*new_column)
            self.selected = 3
            for channel, val in zip(self.hsv_controls, new_column):
                channel.set(round(val, 2))
            self.selected = None
            self.update_cursors(new_column)
            self.update_colour()

        def update_rgb(*_):
            if self.selected is not None:
                return
            try:
                new_column = [float(channel.get()) for channel in self.hsv_controls]
            except ValueError:
                return
            if min(new_column) < 0 or max(new_column) > 1:
                return

            self.selected = 3
            for channel, val in zip(self.rgb_controls, hsv_to_rgb(*new_column)):
                channel.set(round(val, 2))
            self.selected = None
            self.update_cursors(new_column)
            self.update_colour()

        for i, rgb, hsv in zip(range(3), self.rgb_controls, self.hsv_controls):
            tk.Spinbox(self, from_=0, to=1, increment=0.1, width=8, textvariable=rgb).grid(
                row=i + 1, column=0, sticky='e'
            )
            tk.Spinbox(self, from_=0, to=1, increment=0.1, width=8, textvariable=hsv).grid(
                row=i + 1, column=1, sticky='w'
            )

            rgb.trace('w', update_hsv)
            hsv.trace('w', update_rgb)

        tk.Button(self, text='Select', command=self.select).grid(row=5, column=0)
        tk.Button(self, text='Cancel', command=self.cancel).grid(row=5, column=1)

        self.selected = None

        self.update_all(rgb_to_hsv(*initial_colour[:3]))
        self.initial_colour = rgb_to_hsv(*initial_colour[:3])

    def select(self):
        self.destroy()

    def cancel(self):
        self.colour = self.initial_colour.copy()
        self.alpha = self.initial_alpha
        self.update_colour()
        self.destroy()

    def update_colour(self):
        pass

    def click(self, e):
        if abs(e.x - self.size - 10) <= 7 and 10 <= e.y <= self.size - 10:
            self.selected = 0
        elif (e.x - self.size / 2) ** 2 + (e.y - self.size / 2) ** 2 <= (self.size / 2) ** 2:
            self.selected = 1
        else:
            self.selected = None
            return
        self.move(e)

    def release(self, e):
        self.selected = None

    def move(self, e):
        pos = np.array([e.x, e.y], dtype=float)
        centred = pos - self.size / 2
        new_column = self.colour.copy()
        if self.selected == 0:
            self.canvas.delete(self.cursors[1])
            pos[1] = max(10, min(pos[1], self.size - 10))
            self.move_cursor(1, self.size + 10, pos[1])
            val = 1 - (pos[1] - 10) / (self.size - 20)
            self.refresh(val)

            new_column[2] = val
        elif self.selected == 1:
            sat = length(centred) / self.size * 2

            if sat > 1:
                self.move_cursor(0, *(centred * (1 / sat) + self.size / 2))
            else:
                self.move_cursor(0, *pos)

            new_column[0] = (math.atan2(centred[1], centred[0]) / math.pi / 2) % 1
            new_column[1] = min(sat, 1)
        else:
            return
        self.update_textboxes(new_column)
        self.update_colour()

    def create_line(self, col):
        self.canvas.create_line(
            10,
            self.size + 5,
            self.size + 10,
            self.size + 5,
            width=9,
            capstyle=tk.ROUND,
            fill=convert_colour(col),
        )

    def update_all(self, col):
        rgb = hsv_to_rgb(*col)

        for spinbox, val in zip(self.hsv_controls + self.rgb_controls, [*col, *rgb]):
            spinbox.set(round(val, 2))
        self.create_line(rgb)

        self.canvas.delete(self.cursors[0])
        self.canvas.delete(self.cursors[1])
        self.move_cursor(0, *(rotate([col[1] * self.size / 2, 0], col[0] * 2 * math.pi) + self.size / 2))
        self.move_cursor(1, self.size + 10, (1 - col[2]) * (self.size - 20) + 10)

        self.refresh(col[2])

        self.colour = np.array(col)

    def update_textboxes(self, col):
        rgb = hsv_to_rgb(*col)
        self.create_line(rgb)
        for spinbox, val in zip(self.hsv_controls + self.rgb_controls, col + list(rgb)):
            spinbox.set(round(val, 2))
        self.colour = list(col)

    def update_cursors(self, col):
        self.move_cursor(0, *(rotate([col[1] * self.size / 2, 0], col[0] * 2 * math.pi) + self.size / 2))
        self.move_cursor(1, self.size + 10, (1 - col[2]) * (self.size - 20) + 10)

        self.create_line(hsv_to_rgb(*col))

        self.refresh(col[2])

        self.colour = col.copy()

    def refresh(self, val):
        adjusted = (self.base_wheel * 255 * val).astype(int)
        adjusted[self.base_wheel < 0] = 255
        self.img.put(self.format_str % tuple(adjusted.reshape(-1)))

    def move_cursor(self, i, x, y):
        radius = 5
        if self.cursors[i] is not None:
            self.canvas.delete(self.cursors[i])
        self.cursors[i] = self.canvas.create_oval(
            (x - radius, y - radius, x + radius, y + radius), width=2, outline='#101010'
        )


class TriangleManipulator(tk.Frame):
    def __init__(self, master, callback):
        super().__init__(master)

        self.size = 6
        self.callback = callback

        self.camera = np.zeros(2)

        self.canvas = tk.Canvas(self, width=300, height=300, bg='black')
        self.canvas.grid(row=0, column=0)

        self.start = None

        self.point_positions = np.array([[0, 0], [1, 0], [0, 1]], float)

        self.canvas.bind('<Button-1>', self.handle_click)

        def handle_release(_):
            self.canvas.bind('<Motion>', lambda e: None)

        self.canvas.bind('<ButtonRelease-1>', handle_release)

        def change_zoom(factor):
            def func():
                self.size /= factor
                self.redraw()

            return func

        zoom_frame = tk.Frame(self)
        tk.Button(zoom_frame, text='+', command=change_zoom(2)).pack(side='left')
        tk.Button(zoom_frame, text='-', command=change_zoom(0.5)).pack(side='right')
        zoom_frame.grid(row=0, column=0, sticky='se')

        self.redraw()

    def redraw(self):
        self.canvas.delete(*self.canvas.find_all())
        for i in range(1, 30):
            p = i * 6 / 30 - 3
            self.canvas.create_line(
                self.convert_coords_to_px(p, -3),
                self.convert_coords_to_px(p, +3),
                fill='gray',
            )
            self.canvas.create_line(
                self.convert_coords_to_px(-3, p),
                self.convert_coords_to_px(+3, p),
                fill='gray',
            )

        self.canvas.create_line(
            self.convert_coords_to_px(-3, 0),
            self.convert_coords_to_px(3, 0),
            fill='red',
        )

        self.canvas.create_line(
            self.convert_coords_to_px(0, -3),
            self.convert_coords_to_px(0, 3),
            fill='red',
        )

        self.canvas.create_line(
            self.convert_coords_to_px(-3, -3),
            self.convert_coords_to_px(3, -3),
            fill='white',
        )
        self.canvas.create_line(
            self.convert_coords_to_px(3, -3),
            self.convert_coords_to_px(3, 3),
            fill='white',
        )
        self.canvas.create_line(
            self.convert_coords_to_px(3, 3),
            self.convert_coords_to_px(-3, 3),
            fill='white',
        )
        self.canvas.create_line(
            self.convert_coords_to_px(-3, 3),
            self.convert_coords_to_px(-3, -3),
            fill='white',
        )

        for point, next_point in zip(self.point_positions, np.roll(self.point_positions, 1, axis=0)):
            self.canvas.create_line(
                self.convert_coords_to_px(*point),
                self.convert_coords_to_px(*next_point),
                fill='white',
            )

        for point, colour in zip(self.point_positions, ('black', 'red', 'blue')):
            midpoint = self.convert_coords_to_px(*point)
            r = 5
            self.canvas.create_oval(
                (midpoint[0] - r, midpoint[1] - r),
                (midpoint[0] + r, midpoint[1] + r),
                fill=colour,
                outline='white',
            )

    def convert_coords_from_px(self, x: int, y: int) -> np.ndarray:
        pos = (np.array([x, y]) / 300 - 0.5) * self.size
        pos[1] *= -1
        pos += self.camera
        return pos

    def convert_coords_to_px(self, x: float, y: float) -> tuple[int, int]:
        pos = np.array([x, y], dtype=float)
        pos -= self.camera
        pos[1] *= -1
        pos /= self.size
        pos += 0.5
        pos *= 300
        return (int(pos[0]), int(pos[1]))

    def move_point(self, index: int, e):
        destination = self.convert_coords_from_px(e.x, e.y)
        self.point_positions[index] = destination
        self.callback(self.point_positions)
        self.redraw()

    def rotate(self, e):
        d1 = self.convert_coords_from_px(e.x, e.y) - self.start[1][0]
        a1 = math.atan2(d1[0], d1[1])
        d2 = self.start[0] - self.start[1][0]
        a2 = math.atan2(d2[0], d2[1])

        rot = rotation(a1 - a2)
        self.point_positions = (self.start[1] - self.start[1][0]) @ rot + self.start[1][0]

        self.callback(self.point_positions)
        self.redraw()

    def translate(self, e):
        delta = self.convert_coords_from_px(e.x, e.y) - self.start[0]
        self.point_positions = self.start[1] + delta

        self.callback(self.point_positions)
        self.redraw()

    def scale(self, e):
        a, b, c = self.start[1]

        b1, c1 = b - a, c - a
        d = b1 - c1
        rot = rotation(math.atan2(d[1], d[0]))

        scale = ((self.convert_coords_from_px(e.x, e.y) - self.start[1][0]) @ rot)[1] / (b1 @ rot)[1]
        self.point_positions[1] = self.start[1][0] + (self.start[1][1] - self.start[1][0]) * scale
        self.point_positions[2] = self.start[1][0] + (self.start[1][2] - self.start[1][0]) * scale

        self.callback(self.point_positions)
        self.redraw()

    def translate_screen(self, e):
        pos = self.camera + self.start[0] - self.convert_coords_from_px(e.x, e.y)
        self.camera = pos
        self.redraw()

    def handle_click(self, event):
        pos = self.convert_coords_from_px(event.x, event.y)
        for index, point in enumerate(self.point_positions):
            point_px = self.convert_coords_to_px(*point)
            dist2 = (event.x - point_px[0]) ** 2 + (event.y - point_px[1]) ** 2
            if dist2 < 5**2:
                self.canvas.bind('<Motion>', partial(self.move_point, index))
                self.start = pos, self.point_positions.copy()
                return

        for i, (a, b, c) in enumerate(
            zip(
                self.point_positions,
                np.roll(self.point_positions, -1, 0),
                np.roll(self.point_positions, -2, 0),
            )
        ):
            d = b - a
            angle = math.atan2(d[1], d[0])
            rot = rotation(angle)
            n1, n2, n3, p = [v @ rot for v in (a, b, c, pos)]

            if abs(p[1] - n1[1]) < 0.015 * self.size and p[0] > min(n1[0], n2[0]) and p[0] < max(n1[0], n2[0]):
                self.start = pos, self.point_positions.copy()
                if i == 1:
                    self.canvas.bind('<Motion>', self.scale)
                else:
                    self.canvas.bind('<Motion>', self.rotate)
                return

            if (n1[1] > n3[1]) != (n1[1] > p[1]):
                self.start = pos, self.camera.copy()
                self.canvas.bind('<Motion>', self.translate_screen)
                return
        self.start = pos, self.point_positions.copy()
        self.canvas.bind('<Motion>', self.translate)


class MatrixEditor(tk.Toplevel):
    def __init__(self, master, name, matrix):
        super().__init__(master)

        self.name = name
        self.matrix = matrix
        self.initial = copy.copy(matrix)

        self.resizable(False, False)
        master.matrix_editors.append(self)

        self.update_title()

        self.error = tk.StringVar(self)
        self.values = [tk.StringVar(self) for _ in range(6)]

        def create_focus(entry):
            def func(_):
                try:
                    self.update_matrix()
                    self.error.set('')
                    self.update_points()
                except ValueError:
                    entry.after(0, lambda: self.error.set('Invalid Number'))

            return func

        inner = tk.Frame(self)
        for i, val in enumerate(matrix):
            self.values[i].set(self.format_num(val))
            entry = tk.Entry(inner, textvariable=self.values[i], width=8, justify=tk.RIGHT)
            entry.grid(row=i % 2, column=i // 2)
            entry.bind('<FocusOut>', create_focus(entry))
        inner.pack(padx=10, pady=10)

        buttons = tk.Frame(self)

        tk.Button(buttons, text='Restore', command=self.restore).grid(row=0, column=0)
        tk.Button(buttons, text='Random', command=self.rand).grid(row=0, column=1)
        tk.Button(buttons, text='Reset', command=self.reset).grid(row=0, column=2)

        tk.Label(buttons, textvariable=self.error).grid(row=1, column=0, columnspan=3)
        buttons.pack()

        self.manipulator = TriangleManipulator(self, self.set_coefficients)
        self.manipulator.pack(padx=0, pady=0)

        self.update_points()

        self.protocol('WM_DELETE_WINDOW', self.on_close)

    def format_num(self, value):
        return '{:.4f}'.format(value)

    def destroy(self):
        self.master.matrix_editors.remove(self)
        super().destroy()

    def on_close(self):
        try:
            output = [float(value.get()) for value in self.values]
            for i, v in enumerate(output):
                self.matrix[i] = v

            if self.name == 'Pre-Transform':
                self.master.modify(pre_trans=self.matrix)
            elif self.name == 'Post-Transform':
                self.master.modify(post_trans=self.matrix)

            self.destroy()
        except ValueError:
            self.error.set('Invalid Value')

    def update_title(self):
        self.title('{}: {}'.format(self.master.name.get(), self.name))

    def set_coefficients(self, points):
        origin = points[0]
        a = points[1] - origin
        b = points[2] - origin

        self.values[4].set(self.format_num(origin[0]))
        self.values[5].set(self.format_num(origin[1]))

        self.values[0].set(self.format_num(a[0]))
        self.values[1].set(self.format_num(a[1]))
        self.values[2].set(self.format_num(b[0]))
        self.values[3].set(self.format_num(b[1]))

        self.update_matrix()

    def update_points(self):
        self.manipulator.point_positions[0] = float(self.values[4].get()), float(self.values[5].get())
        origin = self.manipulator.point_positions[0]
        self.manipulator.point_positions[1] = origin + np.array(
            [
                float(self.values[0].get()),
                float(self.values[1].get()),
            ]
        )
        self.manipulator.point_positions[2] = origin + np.array(
            [
                float(self.values[2].get()),
                float(self.values[3].get()),
            ]
        )
        self.manipulator.redraw()

    def update_matrix(self):
        output = [float(value.get()) for value in self.values]
        for i, v in enumerate(output):
            self.matrix[i] = v

        if event_queue.empty():
            if self.name == 'Pre-Transform':
                self.master.modify(pre_trans=self.matrix)
            elif self.name == 'Post-Transform':
                self.master.modify(post_trans=self.matrix)

    def reset(self):
        identity = 1, 0, 0, 1, 0, 0
        for var, val in zip(self.values, identity):
            var.set(self.format_num(val))
        self.manipulator.camera[:] = 0
        self.update_points()
        self.update_matrix()

    def restore(self):
        for i, val in enumerate(self.initial):
            self.values[i].set(self.format_num(val))
        self.update_points()
        self.update_matrix()

    def rand(self):
        for val in self.values:
            val.set(self.format_num((random.random() - 0.5) * 4))
        self.update_points()
        self.update_matrix()


def matrix_set(master, title, matrix):
    active = None

    def func():
        nonlocal active
        if active is not None and active.winfo_exists():
            raise_window(active)
            return
        active = MatrixEditor(master, title, matrix)

    return func


class FuncFormatEditor(tk.Toplevel):
    def __init__(self, master, initial=None):
        super().__init__(master)
        self.current = initial


class ScrolledList(tk.Frame):
    def __init__(self, master, width, height):
        super().__init__(master)

        canvas = tk.Canvas(self)
        canvas.pack(side='left', fill=tk.BOTH)
        canvas.configure(scrollregion=canvas.bbox('all'), width=width, height=height)

        def scroll(command, y):
            if command == 'moveto' and float(y) < 0:
                return
            canvas.yview(command, y)

        scrollbar = tk.Scrollbar(self, orient='vertical', command=scroll)
        scrollbar.pack(side='right', fill='y')

        canvas.configure(yscrollcommand=scrollbar.set)

        self.frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.frame, anchor='nw', width=width)

        self.frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

    def insert(self, widget, **settings):
        settings = settings.copy()
        settings['in'] = self.frame
        widget.pack(**settings)


class FunctionEditor(tk.Toplevel):
    def __init__(self, master, func_id, name_var, source):
        super().__init__(master)

        self.name = name_var
        self.function_id = func_id

        self.colour = source['colour']
        if len(self.colour) != 4:
            self.colour += [0.5]
        self.variations = {}
        self.pre_trans = source['pre_trans']
        self.post_trans = source['post_trans']

        for var_data in source['variations']:
            func_format = FunctionFormat(
                var_data['name'],
                var_data['base'],
                list(var_data.get('params', {}).keys()),
                var_data.get('images', {}),
            )
            self.variations[func_format] = list(var_data['params'].values())
        self.modify(format=[], pre_trans=self.pre_trans, post_trans=self.post_trans, colour=self.colour)

        self.picker = None
        self.matrix_editors = []
        # renderer.active_overlays.append(self.function)
        self.resizable(False, False)
        # self.attributes('-toolwindow', 1)
        inner = tk.Frame(self)
        inner.pack(padx=5)
        buttons = tk.Frame(inner)
        buttons.pack(pady=5)

        self.change_title()
        self.titler = self.name.trace_variable('w', self.change_title)

        tk.Entry(buttons, textvariable=self.name).grid(row=0, column=0, columnspan=3, pady=5)

        tk.Button(
            buttons,
            text='Pre Trans',
            command=matrix_set(self, 'Pre-Transform', self.pre_trans),
        ).grid(row=1, column=0)

        self.colour_but = tk.Button(buttons, bg=convert_colour(self.colour[:3]), command=self.colour_pressed)
        self.colour_but.grid(row=1, column=1, ipadx=8)
        tk.Button(
            buttons,
            text='Post Trans',
            command=matrix_set(self, 'Post-Transform', self.post_trans),
        ).grid(row=1, column=2)

        self.error = tk.StringVar(self)
        tk.Label(self, textvariable=self.error).pack()

        self.scrolled = ScrolledList(inner, 220, 350)
        self.scrolled.pack()

        self.variation_frame = None

        self.update_variation_list()
        self.protocol('WM_DELETE_WINDOW', self.on_close)

    def fix_paths(self, func_formats):
        app = self.master.master

        func_formats = copy.deepcopy(func_formats)
        for func_format in func_formats:
            for name in func_format.images:
                filename = func_format.images[name]
                fixed = path.join(app.current_directory, filename)

                func_format.images[name] = fixed
        return func_formats

    def update_variation_list(self):
        self.modify(
            format=self.fix_paths(list(self.variations)),
            params=reduce(operator.iadd, self.variations.values(), []),
        )

        if self.variation_frame is not None:
            self.variation_frame.pack_forget()

        self.variation_frame = tk.Frame(self.scrolled.frame)

        def update_variation(func_format):
            window = tk.Toplevel(self)

            window.title(func_format.name)
            window.resizable(False, False)

            frame = tk.Frame(window)
            frame.pack(padx=5, pady=5)

            header = tk.Frame(frame)
            header.grid(row=0, column=0)

            tk.Label(header, text='Variation Name:').grid(row=0, column=0)
            name_var = tk.StringVar(header, func_format.name)
            tk.Entry(header, textvariable=name_var).grid(row=0, column=1)

            name_var.trace('w', lambda *args: window.title(name_var.get()))

            tk.Label(frame, text=HELPERS, justify=tk.LEFT, font='TkFixedFont').grid(row=1, column=0, sticky='w')

            text_widget = tk.Text(frame)
            text_widget.grid(row=2, column=0)

            text_widget.insert(tk.END, func_format.base)

            properties = tk.Frame(frame)
            properties.grid(row=3, column=0)

            tk.Label(properties, text='Parameter Names:').grid(row=0, column=0, sticky='e')
            parameters_var = tk.StringVar(properties, ', '.join(func_format.params))
            tk.Entry(properties, textvariable=parameters_var, width=60).grid(row=0, column=1, sticky='w')

            tk.Label(properties, text='Images:').grid(row=2, column=0, sticky='e')
            images_var = tk.StringVar(
                properties,
                ', '.join(name + ': ' + filename for name, filename in func_format.images.items()),
            )
            tk.Entry(properties, textvariable=images_var, width=60).grid(row=2, column=1, sticky='w')

            error_var = tk.StringVar()
            tk.Label(frame, textvariable=error_var).grid(row=4, column=0)

            footer = tk.Frame(frame)
            footer.grid(row=5, column=0)

            for var in (parameters_var, images_var):
                var.trace('w', lambda *args: error_var.set(''))

            def parse():
                parameters = []
                for piece in parameters_var.get().split(','):
                    parameter = piece.strip()
                    if len(parameter) == 0:
                        continue
                    if ' ' in parameter:
                        raise ValueError
                    parameters.append(parameter)
                parameters = sorted(set(parameters))

                images = {}
                for piece in images_var.get().split(','):
                    if len(piece.strip()) == 0:
                        continue
                    name, filename = piece.split(':')
                    name = name.strip()
                    filename = filename.strip()

                    if ' ' in name:
                        raise ValueError

                    if name in images:
                        raise ValueError

                    images[name] = filename

                return parameters, images

            def save():
                try:
                    parameters, images = parse()
                except ValueError:
                    error_var.set('Error')
                    print_exc()
                    return

                func_format.name = name_var.get()
                func_format.params = parameters
                func_format.images = images
                func_format.base = text_widget.get(1.0, tk.END)

                self.update_variation_list()

                window.destroy()

            def cancel():
                window.destroy()

            tk.Button(footer, command=save, text='Save').pack(side=tk.LEFT)
            tk.Button(footer, command=cancel, text='Cancel').pack(side=tk.LEFT)

            window.wait_window()

        def delete_variation(func_format):
            del self.variations[func_format]
            self.update_variation_list()

        def update_param(func_format, i, var, *_):
            try:
                params = self.variations[func_format]
                params[i] = var.get()
                self.modify(params=reduce(operator.iadd, self.variations.values(), []))
            except tk.TclError as e:
                print(f'Invalid value: {e}')

        for func_format, values in self.variations.items():
            values = values.copy()
            entry = tk.Frame(self.variation_frame)

            params = func_format.params.copy()
            tk.Label(entry, text=func_format.name, justify=tk.LEFT).grid(row=0, column=0, sticky='ew')
            try:
                weight_i = params.index('weight')
                var = tk.DoubleVar(entry, value=values[weight_i])

                del params[weight_i]
                del values[weight_i]

                var.trace('w', partial(update_param, func_format, weight_i, var))
                spin = tk.Spinbox(entry, textvariable=var, from_=-3, to=3, increment=0.05, repeatdelay=50, width=5)

                spin.grid(row=0, column=1, sticky='e')
            except ValueError:
                weight_i = float('inf')

            tk.Button(entry, text='Del', command=partial(delete_variation, func_format)).grid(row=0, column=2)
            tk.Button(entry, text='Edit', command=partial(update_variation, func_format)).grid(row=0, column=3)

            if len(params) != 0:
                frame = tk.Frame(entry, relief=tk.SUNKEN, borderwidth=1)
                for i, (param, val) in enumerate(zip(params, values)):
                    if i >= weight_i:
                        i += 1
                    var = tk.DoubleVar(frame, value=val)
                    var.trace('w', partial(update_param, func_format, i, var))

                    tk.Label(frame, text=param, justify=tk.RIGHT).grid(row=i, column=0, sticky='e')
                    tk.Spinbox(
                        frame,
                        textvariable=var,
                        from_=-3,
                        to=3,
                        increment=0.05,
                        repeatdelay=50,
                        width=5,
                    ).grid(row=i, column=1, sticky='w')

                frame.grid(row=2, column=0, columnspan=4, ipadx=5, ipady=5)

            entry.pack(expand=1, fill=tk.X)

        def add_variation():
            selector = tk.Toplevel(self)
            selector.title('Select Variation')
            selector.resizable(False, False)
            selector.grab_set()

            scrolled_list = ScrolledList(selector, 150, 400)

            def return_variation(variation):
                nonlocal returned_variation
                returned_variation = variation
                selector.destroy()

            returned_variation = None
            for variation in BASE_FUNCTIONS:
                entry = tk.Button(selector, text=variation.name, command=partial(return_variation, variation))
                scrolled_list.insert(entry, expand=1, fill=tk.X)
            scrolled_list.pack()

            selector.wait_window()

            if returned_variation is None or returned_variation in self.variations:
                return

            variation = copy.deepcopy(returned_variation)

            params = [0] * len(variation.params)
            try:
                i = variation.params.index('weight')
                params[i] = 1
            except ValueError:
                pass
            self.variations[variation] = params

            self.update_variation_list()

        tk.Button(self.variation_frame, text='Add', command=add_variation).pack()

        self.variation_frame.pack()

    def modify(self, **changes):
        modify_func(self.function_id, **changes)

    def colour_pressed(self):
        if self.picker is not None:
            raise_window(self.picker)
            return

        self.picker = ColourPicker(self.colour)

        def on_update_colour():
            self.colour[:3] = hsv_to_rgb(*self.picker.colour)
            self.colour[3] = self.picker.alpha
            self.modify(colour=self.colour)

        self.picker.update_colour = on_update_colour

        picker_destroy = self.picker.destroy

        def on_destroy():
            self.picker = None
            if self.colour_but.winfo_exists():  # Case where everything is being destroyed
                self.colour_but['bg'] = convert_colour(self.colour[:3])
            picker_destroy()

        self.picker.destroy = on_destroy

        self.picker.title(self.name.get())

    def on_close(self):
        picker = self.picker
        if picker is not None:
            picker.destroy()
        for editor in self.matrix_editors:
            editor.destroy()

        focus = self.focus_get()
        if type(focus) is tk.Entry:
            self.focus_set()

        self.withdraw()

    def destroy(self):
        self.name.trace_vdelete('w', self.titler)
        super().destroy()

    def change_title(self, *_):
        self.title('Edit ' + self.name.get())
        picker = self.picker
        if picker is not None:
            picker.update_title()
        for editor in self.matrix_editors:
            editor.update_title()


class FunctionWidget(tk.Frame):
    def __init__(self, master, source, func_id):
        super().__init__(master)

        self.function_id = func_id

        modify_func(func_id, format=[], prob=source['prob'])

        self.name = tk.StringVar(self, value=source['name'])
        self.menu = FunctionEditor(self, self.function_id, self.name, source)
        self.menu.withdraw()

        tk.Label(self, textvariable=self.name, width=11).grid(row=0, column=0)
        tk.Button(self, text='Edit', command=self.edit).grid(row=0, column=1)
        tk.Button(self, text='Del', command=self.delete).grid(row=0, column=2)

        prob = source['prob']
        if isinstance(prob, list):
            prob = 0
        self.prob = tk.DoubleVar(self, value=prob)

        self.prob._set, self.prob.set = self.prob.set, lambda value: self.prob._set('{:.3f}'.format(value))
        # self.prob.set(function.prob)

        def slide(_):
            widgets = self.master.get_functions()
            if len(widgets) == 1:
                self.prob.set(1)
            else:
                if self.prob.get() > 1:
                    self.prob.set(1)

                value = self.prob.get()
                others = sum(func.prob.get() for func in widgets if func is not self)

                last_widget = len(widgets) - 1

                if others == 0:
                    factor = (1 - value) / last_widget
                    for func in widgets:
                        if func is self:
                            continue
                        func.prob.set(func.prob.get() + factor)
                else:
                    factor = (1 - value) / others
                    for func in widgets:
                        if func is self:
                            continue
                        func.prob.set(round(func.prob.get() * factor, 3))

        frame = tk.Frame(self)
        frame.grid(row=1, column=0, columnspan=3)

        tk.Label(frame, textvariable=self.prob).grid(row=0, column=0, sticky='w')

        self.slider = tk.Scale(
            frame,
            from_=0,
            to=1,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            showvalue=False,
            variable=self.prob,
            command=slide,
            length=150,
        )
        self.transition_probs = object()

        if isinstance(source['prob'], list):
            self.set_transition_probs(source['prob'])
        else:
            self.set_transition_probs(None)

        def change_prob(*_):
            if self.transition_probs is None:
                modify_func(func_id, prob=self.prob.get())

        self.prob.trace_variable('w', change_prob)

    def set_transition_probs(self, transition_probs):
        if transition_probs == self.transition_probs:
            return

        self.transition_probs = transition_probs

        if transition_probs is None:
            self.slider.grid(row=0, column=1)
            modify_func(self.function_id, prob=self.prob.get())
        else:
            self.slider.grid_forget()
            modify_func(self.function_id, prob=transition_probs)

    def delete(self):
        functions = self.master.get_functions()

        i = functions.index(self)
        self.destroy()
        del functions[i]

        delete_func(self.function_id)

        if len(functions) != 0:
            if self.transition_probs is None:
                for func, prob in zip(functions, fix_probabilities([func.prob.get() for func in functions])):
                    func.prob.set(prob)
            else:
                for func in functions:
                    transition_probs = func.transition_probs.copy()
                    del transition_probs[i]
                    func.set_transition_probs(fix_probabilities(transition_probs))

                self.master.update_approx_probs()

    def edit(self):
        self.menu.deiconify()


class MarkovMatrixEditor(tk.Toplevel):
    def __init__(self, master, func_names, transitions, initially_enabled, on_change):
        super().__init__(master)

        self.on_change = on_change

        self.enabled = tk.BooleanVar(self, initially_enabled)
        self.enabled.trace('w', self.update_enabled)

        self.transitions = transitions
        self.initial = initially_enabled, copy.deepcopy(transitions)

        self.title('Edit Transition Probabilities')
        self.resizable(False, False)

        tk.Checkbutton(self, text='Enable Full Markov Chain', variable=self.enabled).pack(padx=4, pady=4)

        inner = tk.Frame(self)
        inner.pack(padx=4, pady=4)

        self._modifying_spinboxes = True

        self.spinboxes = []
        self.variables = []
        for y, row in enumerate(transitions):
            var_row = []

            tk.Label(inner, text=func_names[y] + ':').grid(row=y, column=0, sticky='e')

            for x, prob in enumerate(row):
                variable = tk.StringVar(inner, '%.3f' % prob)
                variable.trace('w', partial(self.modify, y, x))
                var_row.append(variable)

                spinbox = tk.Spinbox(
                    inner,
                    textvariable=variable,
                    from_=0,
                    to=1,
                    increment=0.005,
                    repeatdelay=50,
                    width=5,
                )
                if not self.enabled.get():
                    spinbox['state'] = tk.DISABLED
                spinbox.grid(row=y, column=x + 1)
                self.spinboxes.append(spinbox)

            self.variables.append(var_row)

        self._modifying_spinboxes = False

        tk.Button(self, text='Restore', command=self.restore).pack()

    def update_enabled(self, *_):
        enabled = self.enabled.get()
        for spinbox in self.spinboxes:
            spinbox['state'] = tk.NORMAL if enabled else tk.DISABLED

        if enabled:
            self.on_change(True, self.transitions)
        else:
            self.on_change(False, None)

    def modify(self, y, x, *_):
        if self._modifying_spinboxes:
            return

        row = self.transitions[y]
        var_row = self.variables[y]
        var = var_row[x]

        try:
            val = float(var.get())
        except ValueError:
            return

        try:
            self._modifying_spinboxes = True

            if len(row) == 1:
                row[0] = 1
                var.set('%.3f' % 1)
                return

            if val < 0:
                var.set('%.3f' % 0)
                val = 0
            elif val > 1:
                var.set('%.3f' % 1)
                val = 1

            row[x] = 0
            total = sum(row)
            row[x] = val

            last_row = len(row) - 1

            if total == 0:
                v = (1 - val) / last_row
                for i, other_var in enumerate(var_row):
                    if i == x:
                        continue
                    row[i] = v
                    other_var.set(v)
            else:
                factor = (1 - val) / total
                for i, other_var in enumerate(var_row):
                    if i == x:
                        continue
                    row[i] *= factor
                    other_var.set('%.3f' % row[i])

            assert self.enabled.get()
            self.on_change(True, self.transitions)
        finally:
            self._modifying_spinboxes = False

    def restore(self):
        initially_enabled, transitions = self.initial

        self.transitions = copy.deepcopy(transitions)
        self.enabled.set(initially_enabled or self.enabled.get())

        try:
            self._modifying_spinboxes = True
            for row, var_row in zip(transitions, self.variables):
                for prob, var in zip(row, var_row):
                    var.set('%.3f' % prob)
        finally:
            self._modifying_spinboxes = False

        if self.enabled.get():
            self.on_change(True, self.transitions)
        else:
            self.on_change(True, self.transitions)  # Hmmm
            self.on_change(False, None)


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.current_directory = path.dirname(__file__)
        self.uses_markov_chain = False

        self.title('Flame Confrigulator')
        self.resizable(False, False)

        self.current_id = 0

        self.gamma = 2
        self.hits_per_tick = 150
        update_properties(self.gamma, self.hits_per_tick)

        self.radius = 6
        self.position = [0, 0]
        update_view(self.position, self.radius)

        inner = tk.Frame(self)
        inner.pack(padx=5, pady=5)

        save_load = tk.Frame(inner)
        save_load.grid(row=0)
        tk.Button(save_load, text='Reset', command=self.do_reset).pack(side='right')
        tk.Button(save_load, text='Save', command=self.do_save).pack(side='right')
        tk.Button(save_load, text='Load', command=self.do_load).pack(side='right')

        scrolled = ScrolledList(inner, 200, 400)
        scrolled.grid(row=2, pady=5)

        self.func_list = tk.Frame()
        scrolled.insert(self.func_list)
        scrolled.insert(tk.Button(text='Create', command=self.create_function))

        tk.Button(inner, text='Markov Chain Weights', command=self.show_markov_chain_editor).grid(row=3, sticky='ew')
        ttk.Separator(inner).grid(row=4, sticky='ew', pady=3)
        tk.Button(inner, text='Render', command=reset).grid(row=5, sticky='ew')
        tk.Button(inner, text='Render Settings...', command=self.render_settings).grid(row=6, sticky='ew')
        tk.Button(inner, text='Save Image', command=self.save_image).grid(row=7, sticky='ew')

    def show_markov_chain_editor(self):
        functions = self.get_functions()

        if len(functions) < 2:
            messagebox.showerror('Invalid Usage', 'Add some functions first')
            return

        func_names = [func.name.get() for func in functions]

        if self.uses_markov_chain:
            probabilities = [func.transition_probs.copy() for func in functions]
        else:
            probabilities = [[func.prob.get() for func in functions] for _ in functions]

        def on_change(use_markov_chain, transitions=None):
            self.uses_markov_chain = use_markov_chain
            set_use_markov_chain(self.uses_markov_chain)
            if use_markov_chain:
                for i, func in enumerate(functions):
                    vec = np.zeros(len(functions))
                    vec[i] = 1

                    probs = np.array(transitions[i])
                    func.set_transition_probs((probs / np.sum(probs)).tolist())

                self.update_approx_probs()

            else:
                for func in functions:
                    func.set_transition_probs(None)

        editor = MarkovMatrixEditor(self, func_names, probabilities, self.uses_markov_chain, on_change)
        editor.transient(self)
        editor.grab_set()
        editor.wait_window()

    def update_approx_probs(self):
        assert self.uses_markov_chain
        functions = self.get_functions()

        transitions = np.array([func.transition_probs for func in functions], dtype=float).T
        approx_prob = approximate_markov_steady_state(transitions)

        for func, prob in zip(functions, approx_prob):
            func.prob.set(prob)

    def create_function(self):
        popup = tk.Toplevel(self)
        popup.resizable(False, False)
        popup.grab_set()

        name = tk.StringVar()
        tk.Label(popup, text='Enter Name').pack()
        tk.Entry(popup, textvariable=name).pack(padx=10)

        value = None

        def finish():
            nonlocal value
            value = name.get()
            if value == '':
                i = 0
                for function in self.get_functions():
                    match = re.match('^Function ([0-9]+)$', function.name.get())
                    if match is not None:
                        i = max(i, int(match[1]))
                value = 'Function ' + str(i + 1)
            popup.destroy()

        tk.Button(popup, text='Done', command=finish).pack(pady=5)

        popup.wait_window()

        if value is None:
            return

        func_id = self.current_id
        self.current_id += 1

        source = {
            'name': value,
            'colour': [1, 1, 1, 0.5],
            'variations': [],
            'pre_trans': [1, 0, 0, 1, 0, 0],
            'post_trans': [1, 0, 0, 1, 0, 0],
        }

        initial_functions = self.get_functions()

        if self.uses_markov_chain:
            source['prob'] = [0] * len(initial_functions) + [1]
        elif len(initial_functions) == 0:
            source['prob'] = 1
        else:
            source['prob'] = 0

        widget = FunctionWidget(self, source, func_id)
        widget.pack(in_=self.func_list)

        if self.uses_markov_chain:
            for func in initial_functions:
                func.transition_probs.append(0)

            self.update_approx_probs()

    def get_functions(self):
        return self.func_list.pack_slaves()

    def render_settings(self):
        popup = tk.Toplevel()
        popup.title('Settings')
        popup.resizable(False, False)
        inner = tk.Frame(popup)
        inner.pack(padx=5, pady=5)

        x, y, r, g, h = variables = [
            tk.StringVar(value=str(val)) for val in (*self.position, self.radius, self.gamma, self.hits_per_tick)
        ]

        def change_view(*_):
            try:
                pos_x = float(x.get())
                pos_y = float(y.get())
                radius = float(r.get())

                self.position[:] = pos_x, pos_y
                self.radius = radius

                update_view(self.position, self.radius)
            except ValueError:
                error.set('Invalid Number')

        def change_properties(*_):
            try:
                gamma = float(g.get())
                hits_per_tick = float(h.get())

                self.gamma = gamma
                self.hits_per_tick = hits_per_tick

                update_properties(self.gamma, self.hits_per_tick)
                error.set('')
            except ValueError:
                error.set('Invalid Number')

        for var in variables[:3]:
            var.trace('w', change_view)

        for var in variables[3:5]:
            var.trace('w', change_properties)

        tk.Label(inner, text='Centre:').grid(row=0, column=0, sticky='e')
        tk.Label(inner, text='Radius:').grid(row=1, column=0, sticky='e')
        tk.Label(inner, text='Gamma:').grid(row=2, column=0, sticky='e')
        tk.Label(inner, text='Hit Rate:').grid(row=3, column=0, sticky='e')

        tk.Entry(inner, width=6, textvariable=x).grid(row=0, column=1)
        tk.Entry(inner, width=6, textvariable=y).grid(row=0, column=2)
        tk.Entry(inner, width=10, textvariable=r).grid(row=1, column=1, columnspan=2, sticky='w')
        tk.Entry(inner, width=10, textvariable=g).grid(row=2, column=1, columnspan=2, sticky='w')
        tk.Entry(inner, width=10, textvariable=h).grid(row=3, column=1, columnspan=2, sticky='w')

        error = tk.StringVar()
        tk.Label(inner, textvariable=error).grid(row=4, column=0, columnspan=3)

    def save_image(self):
        filename = filedialog.asksaveasfilename(
            title='Save Render',
            defaultextension='.png',
            filetypes=(('PNG files', '*.png'), ('JPEG files', '*.jpg'), ('All files', '*.*')),
        )
        if filename == '':
            return
        save_image(filename)

    def do_save(self):
        filename = filedialog.asksaveasfilename(
            title='Save Flame',
            defaultextension='.json',
            filetypes=(('Flame files', '*.json'), ('All files', '*.*')),
        )
        if filename == '':
            return

        self.current_directory = path.dirname(filename)

        functions = []
        for func in self.get_functions():
            menu = func.menu
            variations = []

            for func_format, param_values in menu.variations.items():
                variations.append(
                    {
                        'name': func_format.name,
                        'base': func_format.base,
                        'params': dict(zip(func_format.params, param_values)),
                        'images': func_format.images,
                    }
                )

            encoded = {
                'name': func.name.get(),
                'prob': func.prob.get(),
                'colour': menu.colour,
                'pre_trans': menu.pre_trans,
                'post_trans': menu.post_trans,
                'variations': variations,
            }

            if self.uses_markov_chain:
                encoded['prob'] = func.transition_probs

            functions.append(encoded)

        content = {
            'pos': self.position,
            'radius': self.radius,
            'gamma': self.gamma,
            'hits_per_tick': self.hits_per_tick,
            'functions': functions,
        }
        with open(filename, 'w') as f:
            json.dump(content, f)

    def do_load(self):
        filename = filedialog.askopenfilename(
            title='Load Flame',
            filetypes=(('Flame files', '*.json'), ('All files', '*.*')),
        )
        if type(filename) is tuple or filename == '':
            return

        self.load(filename)

    def load(self, filename: str):
        with open(filename, 'r') as f:
            content = json.load(f)

        self.current_directory = path.dirname(filename)

        self.do_reset()

        self.gamma = content['gamma']
        self.hits_per_tick = content['hits_per_tick']
        update_properties(self.gamma, self.hits_per_tick)

        self.position[:] = content['pos']
        self.radius = content['radius']
        update_view(self.position, self.radius)

        self.uses_markov_chain = isinstance(content['functions'][0]['prob'], list)
        set_use_markov_chain(self.uses_markov_chain)

        for func_data in content['functions']:
            func_id = self.current_id
            self.current_id += 1

            widget = FunctionWidget(self, func_data, func_id)
            widget.pack(in_=self.func_list)

            colour = func_data['colour']
            if len(colour) < 4:
                colour.append(0.5)
            widget.menu.colour[:] = colour
            widget.menu.colour_but['bg'] = convert_colour(func_data['colour'][:3])

        if self.uses_markov_chain:
            self.update_approx_probs()

        reset()

    def do_reset(self):
        for widget in self.get_functions():
            widget.destroy()
            delete_func(widget.function_id)


def set_use_markov_chain(enabled):
    event_queue.put(('markov_chain', enabled))


def modify_func(func_id, **changes):
    event_queue.put(('func_change', (func_id, changes)))


def delete_func(func_id):
    event_queue.put(('func_change', (func_id, None)))


def update_view(position, radius):
    event_queue.put(('view_change', (position, radius)))


def update_properties(gamma, hits_per_tick):
    event_queue.put(('properties_change', (gamma, hits_per_tick)))


def reset():
    event_queue.put(('reset', None))


def save_image(filename):
    event_queue.put(('save_image', filename))


def run_render(
    event_queue,
    size,
    supersamples,
    max_iterations,
    target_fps,
    backend_name: BackendMethod,
):
    import OpenGL.GL as GL
    import pygame
    import pygame.locals as pg_locals

    from flames.render import render_to_file
    from flames.renderer import FlameRenderer

    pygame.display.set_mode(size, pg_locals.OPENGL)  # |pg_locals.DOUBLEBUF)

    iterator = create_flame_iterator(backend_name, size, supersamples)
    renderer = FlameRenderer(iterator.colour, iterator.histogram)

    clock = pygame.time.Clock()

    cycles = 100

    gamma = 1
    hits_per_tick = 1

    max_ticks = max_iterations // iterator.particles

    functions = {}

    title = 'Fractal Flames'
    pygame.display.set_caption(title)

    end_time = None
    start_time = time.time()

    recently_updated = 0

    # query = glGenQueries(1)

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pg_locals.QUIT:
                    running = False
            try:
                for _ in range(10):
                    name, data = event_queue.get_nowait()
                    if name == 'reset':
                        iterator.reset_all()
                        start_time = time.time()
                        end_time = None
                    elif name == 'markov_chain':
                        iterator.set_use_markov_chain(data)
                    elif name == 'func_change':
                        func_id, change = data
                        if change is None:
                            func = functions.pop(func_id)
                            del iterator.functions[func]
                        else:
                            if 'format' in change:
                                old_func = functions.pop(func_id, None)
                                if old_func is not None:
                                    old_prob = iterator.functions.pop(old_func)

                                functions[func_id] = func = iterator.create_function(change['format'])

                                if old_func is not None:
                                    iterator.functions[func] = old_prob
                                    func.colour[:] = old_func.colour
                                    func.pre_trans[:] = old_func.pre_trans
                                    func.post_trans[:] = old_func.post_trans
                            else:
                                func = functions[func_id]

                            if 'prob' in change:
                                iterator.functions[func] = change['prob']

                            if 'params' in change:
                                func.params[:] = change['params']

                            if 'colour' in change:
                                func.colour[:] = change['colour']

                            if 'pre_trans' in change:
                                func.pre_trans[:] = change['pre_trans']

                            if 'post_trans' in change:
                                func.post_trans[:] = change['post_trans']
                            recently_updated = 3

                        iterator.reset()
                        start_time = time.time()
                        end_time = None
                    elif name == 'view_change':
                        iterator.centre[:], iterator.radius = data
                        iterator.reset()
                        start_time = time.time()
                        end_time = None
                    elif name == 'properties_change':
                        gamma, hits_per_tick = data
                    elif name == 'save_image':
                        maximum_val = (
                            hits_per_tick * iterator.tick * iterator.particles / (iterator.size[0] * iterator.size[1])
                        )
                        with render_to_file(data, iterator.size.tolist(), False):
                            renderer.render(int(maximum_val), gamma)
                    elif name == 'quit':
                        running = False
            except queue.Empty:
                pass

            if iterator.tick < max_ticks:
                if len(iterator.functions) != 0:
                    if recently_updated > 0 and target_fps < 60:
                        recently_updated -= 1
                        cycles += cycles * (clock.get_fps() / 60 - 1) / 40

                        current_cycles = cycles * (target_fps / 60)
                        if current_cycles < 25:
                            current_cycles = 25
                            cycles = 75
                    else:
                        cycles += cycles * (clock.get_fps() / target_fps - 1) / 40
                        current_cycles = cycles

                    cycles = min(cycles, 1000)
                    # glBeginQuery(GL_TIME_ELAPSED, query)
                    current_cycles = round(current_cycles)
                else:
                    current_cycles = 100

                for _ in range(current_cycles):
                    iterator.update()
                    if iterator.tick >= max_ticks:
                        break
                # glEndQuery(GL_TIME_ELAPSED)

                # val = ctypes.c_uint64(0)
                # res = glGetQueryObjectui64v(query, GL_QUERY_RESULT, ctypes.pointer(val))

                # dt = val.value / 1_000_000_000
                # if len(iterator.functions) != 0:
                #    pass
                rate = clock.get_fps() * iterator.particles * current_cycles
                pygame.display.set_caption(
                    '{}: {:.2f} {:,} {}/{}'.format(title, clock.get_fps(), rate, iterator.tick, max_ticks),
                )
            else:
                if end_time is None:
                    end_time = time.time()
                    # cycles = min(max(cycles, 10), 100)
                pygame.display.set_caption('{}: Done {:.2f}s'.format(title, end_time - start_time))

            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            maximum_val = hits_per_tick * iterator.tick * iterator.particles / (iterator.size[0] * iterator.size[1])
            renderer.render(int(maximum_val), gamma)

            pygame.display.flip()
            clock.tick(60)
    finally:
        pygame.quit()


def wait_for_render(render_process: mp.Process, root: App):
    render_process.join()
    root.quit()


def main():
    parser = argparse.ArgumentParser(description='Flame creator')

    parser.add_argument('--width', type=int, default=1000, help='Rendered height')
    parser.add_argument('--height', type=int, default=1000, help='Rendered width')
    parser.add_argument('--supersamples', type=int, default=2, help='Supersamples to perform')
    parser.add_argument('--iterations', type=int, default=10_000_000_000, help='Number of iterations to perform')
    parser.add_argument('--fps', type=int, default=20, help='Target frames per second')
    parser.add_argument('--backend', type=str, default='basic', help='Flame Iterator to use')
    parser.add_argument('file', type=str, nargs='?', help='Initial file to load')

    args = parser.parse_args()

    if args.width <= 0:
        raise RuntimeError('width <= 0')
    if args.height <= 0:
        raise RuntimeError('height <= 0')
    if args.supersamples <= 0:
        raise RuntimeError('supersamples <= 0')
    if args.iterations <= 0:
        raise RuntimeError('iterations <= 0')
    if args.fps <= 0:
        raise RuntimeError('fps <= 0')

    render_process = mp.Process(
        target=run_render,
        args=(event_queue, (args.width, args.height), args.supersamples, args.iterations, args.fps, str(args.backend)),
        name='Render Thread',
    )
    render_process.start()
    exit_wait_thread = None

    try:
        root = App()
        if args.file is not None:
            root.load(args.file)

        exit_wait_thread = threading.Thread(target=wait_for_render, args=(render_process, root))
        exit_wait_thread.start()

        root.mainloop()
    finally:
        event_queue.put(('quit', None))
        if exit_wait_thread is not None:
            exit_wait_thread.join()


if __name__ == '__main__':
    main()
