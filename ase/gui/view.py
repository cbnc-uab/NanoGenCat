from __future__ import division
from math import cos, sin, sqrt
from os.path import basename

import numpy as np

from ase.data import chemical_symbols
from ase.data.colors import jmol_colors
from ase.geometry import complete_cell
from ase.gui.repeat import Repeat
from ase.gui.rotate import Rotate
from ase.gui.render import Render
from ase.gui.colors import ColorWindow
from ase.utils import rotate


GREEN = '#DDFFDD'


class View:
    def __init__(self, rotations):
        self.colormode = 'jmol'  # The default colors
        self.nselected = 0
        self.labels = None
        self.axes = rotate(rotations)
        self.configured = False
        self.frame = None

    def set_coordinates(self, frame=None, focus=None):
        if frame is None:
            frame = self.frame
        self.make_box()
        self.bind(frame)
        n = self.images.natoms
        self.X = np.empty((n + len(self.B1) + len(self.bonds), 3))
        self.set_frame(frame, focus=focus, init=True)

    def set_frame(self, frame=None, focus=False, init=False):
        if frame is None:
            frame = self.frame

        n = self.images.natoms

        if self.frame is not None and self.frame > self.images.nimages:
            self.frame = self.images.nimages - 1

        if init or frame != self.frame:
            A = self.images.A
            nc = len(self.B1)
            nb = len(self.bonds)

            if init or (A[frame] != A[self.frame]).any():
                self.X[n:n + nc] = np.dot(self.B1, A[frame])
                self.B = np.empty((nc + nb, 3))
                self.B[:nc] = np.dot(self.B2, A[frame])

            if nb > 0:
                P = self.images.P[frame]
                Af = self.images.repeat[:, np.newaxis] * A[frame]
                a = P[self.bonds[:, 0]]
                b = P[self.bonds[:, 1]] + np.dot(self.bonds[:, 2:], Af) - a
                d = (b**2).sum(1)**0.5
                r = 0.65 * self.images.r
                x0 = (r[self.bonds[:, 0]] / d).reshape((-1, 1))
                x1 = (r[self.bonds[:, 1]] / d).reshape((-1, 1))
                self.X[n + nc:] = a + b * x0
                b *= 1.0 - x0 - x1
                b[self.bonds[:, 2:].any(1)] *= 0.5
                self.B[nc:] = self.X[n + nc:] + b

            filenames = self.images.filenames
            filename = filenames[frame]
            if (self.frame is None or
                filename != filenames[self.frame] or
                filename is None):
                if filename is None:
                    filename = 'ase.gui'
            filename = basename(filename)
            self.window.title = filename

        self.frame = frame
        self.X[:n] = self.images.P[frame]
        self.R = self.X[:n]
        if focus:
            self.focus()
        else:
            self.draw()

    def set_colors(self):
        self.colormode = 'jmol'
        self.colors = {}
        for z in np.unique(self.images.Z):
            rgb = jmol_colors[z]
            self.colors[z] = ('#{0:02X}{1:02X}{2:02X}'
                              .format(*(int(x * 255) for x in rgb)))

    def make_box(self):
        if not self.window['toggle-show-unit-cell']:
            self.B1 = self.B2 = np.zeros((0, 3))
            return

        V = self.images.A[0]
        nn = []
        for c in range(3):
            v = V[c]
            d = sqrt(np.dot(v, v))
            if d < 1e-12:
                n = 0
            else:
                n = max(2, int(d / 0.3))
            nn.append(n)
        self.B1 = np.zeros((2, 2, sum(nn), 3))
        self.B2 = np.zeros((2, 2, sum(nn), 3))
        n1 = 0
        for c, n in enumerate(nn):
            n2 = n1 + n
            h = 1.0 / (2 * n - 1)
            R = np.arange(n) * (2 * h)

            for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                self.B1[i, j, n1:n2, c] = R
                self.B1[i, j, n1:n2, (c + 1) % 3] = i
                self.B1[i, j, n1:n2, (c + 2) % 3] = j
            self.B2[:, :, n1:n2] = self.B1[:, :, n1:n2]
            self.B2[:, :, n1:n2, c] += h
            n1 = n2
        self.B1.shape = (-1, 3)
        self.B2.shape = (-1, 3)

    def bind(self, frame):
        if not self.window['toggle-show-bonds']:
            self.bonds = np.empty((0, 5), int)
            return

        from ase.atoms import Atoms
        from ase.neighborlist import NeighborList
        nl = NeighborList(self.images.r * 1.5, skin=0, self_interaction=False)
        nl.update(Atoms(positions=self.images.P[frame],
                        cell=(self.images.repeat[:, np.newaxis] *
                              self.images.A[frame]),
                        pbc=self.images.pbc))
        nb = nl.nneighbors + nl.npbcneighbors

        bonds = np.empty((nb, 5), int)
        self.coordination = np.zeros((self.images.natoms), dtype=int)
        if nb == 0:
            return

        n1 = 0
        for a in range(self.images.natoms):
            indices, offsets = nl.get_neighbors(a)
            self.coordination[a] += len(indices)
            for a2 in indices:
                self.coordination[a2] += 1
            n2 = n1 + len(indices)
            bonds[n1:n2, 0] = a
            bonds[n1:n2, 1] = indices
            bonds[n1:n2, 2:] = offsets
            n1 = n2

        i = bonds[:n2, 2:].any(1)
        pbcbonds = bonds[:n2][i]
        bonds[n2:, 0] = pbcbonds[:, 1]
        bonds[n2:, 1] = pbcbonds[:, 0]
        bonds[n2:, 2:] = -pbcbonds[:, 2:]
        self.bonds = bonds

    def toggle_show_unit_cell(self, key=None):
        self.set_coordinates()

    def show_labels(self):
        index = self.window['show-labels']
        if index == 0:
            self.labels = None
        elif index == 1:
            self.labels = ([list(range(self.images.natoms))] *
                           self.images.nimages)
        elif index == 2:
            self.labels = self.images.M
        else:
            self.labels = [[chemical_symbols[x]
                            for x in self.images.Z]] * self.images.nimages

        self.draw()

    def toggle_show_axes(self, key=None):
        self.draw()

    def toggle_show_bonds(self, key=None):
        self.set_coordinates()

    def toggle_show_velocities(self, key=None):
        # XXX hard coded scale is ugly
        self.show_vectors(10 * np.nan_to_num(self.images.V))
        self.draw()

    def toggle_show_forces(self, key=None):
        self.show_vectors(np.nan_to_num(self.images.F))
        self.draw()

    def hide_selected(self):
        self.images.visible[self.images.selected] = False
        self.draw()

    def show_selected(self):
        self.images.visible[self.images.selected] = True
        self.draw()

    def repeat_window(self, key=None):
        Repeat(self)

    def rotate_window(self):
        return Rotate(self)

    def colors_window(self, key=None):
        win = ColorWindow(self)
        self.register_vulnerable(win)
        return win

    def focus(self, x=None):
        cell = (self.window['toggle-show-unit-cell'] and
                self.images.A[0].any())
        if (self.images.natoms == 0 and not cell):
            self.scale = 1.0
            self.center = np.zeros(3)
            self.draw()
            return

        P = np.dot(self.X, self.axes)
        n = self.images.natoms
        P[:n] -= self.images.r[:, None]
        P1 = P.min(0)
        P[:n] += 2 * self.images.r[:, None]
        P2 = P.max(0)
        self.center = np.dot(self.axes, (P1 + P2) / 2)
        S = 1.3 * (P2 - P1)
        w, h = self.window.size
        if S[0] * h < S[1] * w:
            self.scale = h / S[1]
        elif S[0] > 0.0001:
            self.scale = w / S[0]
        else:
            self.scale = 1.0
        self.draw()

    def reset_view(self, menuitem):
        self.axes = rotate('0.0x,0.0y,0.0z')
        self.set_coordinates()
        self.focus(self)

    def set_view(self, key):
        if key == 'Z':
            self.axes = rotate('0.0x,0.0y,0.0z')
        elif key == 'X':
            self.axes = rotate('-90.0x,-90.0y,0.0z')
        elif key == 'Y':
            self.axes = rotate('90.0x,0.0y,90.0z')
        elif key == 'Alt+Z':
            self.axes = rotate('180.0x,0.0y,90.0z')
        elif key == 'Alt+X':
            self.axes = rotate('0.0x,90.0y,0.0z')
        elif key == 'Alt+Y':
            self.axes = rotate('-90.0x,0.0y,0.0z')
        else:
            if key == '3':
                i, j = 0, 1
            elif key == '1':
                i, j = 1, 2
            elif key == '2':
                i, j = 2, 0
            elif key == 'Alt+3':
                i, j = 1, 0
            elif key == 'Alt+1':
                i, j = 2, 1
            elif key == 'Alt+2':
                i, j = 0, 2

            A = complete_cell(self.images.A[self.frame])
            x1 = A[i]
            x2 = A[j]

            norm = np.linalg.norm

            x1 = x1 / norm(x1)
            x2 = x2 - x1 * np.dot(x1, x2)
            x2 /= norm(x2)
            x3 = np.cross(x1, x2)

            self.axes = np.array([x1, x2, x3]).T

        self.set_coordinates()

    def get_colors(self, rgb=False):
        if rgb:
            return [tuple(int(rgb[i:i + 2], 16) / 255 for i in range(1, 7, 2))
                    for rgb in self.get_colors()]

        if self.colormode == 'jmol':
            return [self.colors[Z] for Z in self.images.Z]

        scalars = self.get_color_scalars()
        colorscale, cmin, cmax = self.colormode_data
        N = len(colorscale)
        indices = np.clip(((scalars - cmin) / (cmax - cmin) * N +
                           0.5).astype(int),
                          0, N - 1)
        return [colorscale[i] for i in indices]

    def get_color_scalars(self, frame=None):
        i = frame or self.frame

        if self.colormode == 'tag':
            return self.images.T[i]
        if self.colormode == 'force':
            f = (self.images.F[i]**2).sum(1)**0.5
            return f * self.images.dynamic
        elif self.colormode == 'velocity':
            return (self.images.V[i]**2).sum(1)**0.5
        elif self.colormode == 'charge':
            return self.images.q[i]
        elif self.colormode == 'magmom':
            return self.images.M[i]

    def draw(self, status=True):
        self.window.clear()
        axes = self.scale * self.axes * (1, -1, 1)
        offset = np.dot(self.center, axes)
        offset[:2] -= 0.5 * self.window.size
        X = np.dot(self.X, axes) - offset
        n = self.images.natoms
        self.indices = X[:, 2].argsort()
        if self.window['toggle-show-bonds']:
            r = self.images.r * (0.65 * self.scale)
        else:
            r = self.images.r * self.scale
        P = self.P = X[:n, :2]
        A = (P - r[:, None]).round().astype(int)
        X1 = X[n:, :2].round().astype(int)
        X2 = (np.dot(self.B, axes) - offset).round().astype(int)
        disp = (np.dot(self.images.D[self.frame], axes)).round().astype(int)
        d = (2 * r).round().astype(int)

        vectors = (self.window['toggle-show-velocities'] or
                   self.window['toggle-show-forces'])
        if vectors:
            V = np.dot(self.vectors[self.frame], axes) + X[:n]

        colors = self.get_colors()
        circle = self.window.circle
        line = self.window.line
        dynamic = self.images.dynamic
        selected = self.images.selected
        visible = self.images.visible
        ncell = len(self.B1)
        bw = self.scale * 0.15
        for a in self.indices:
            if a < n:
                ra = d[a]
                if visible[a]:
                    # Draw the atoms
                    if self.moving and selected[a]:
                        circle(GREEN, False,
                               A[a, 0] - 4, A[a, 1] - 4,
                               A[a, 0] + ra + 4, A[a, 1] + ra + 4)

                    circle(colors[a], selected[a],
                           A[a, 0], A[a, 1], A[a, 0] + ra, A[a, 1] + ra)

                    # Draw labels on the atoms
                    if self.labels is not None:
                        self.window.text(A[a, 0], A[a, 1],
                                         str(self.labels[self.frame][a]))

                    # Draw cross on constrained atoms
                    if not dynamic[a]:
                        R1 = int(0.14644 * ra)
                        R2 = int(0.85355 * ra)
                        line((A[a, 0] + R1, A[a, 1] + R1,
                              A[a, 0] + R2, A[a, 1] + R2))
                        line((A[a, 0] + R2, A[a, 1] + R1,
                              A[a, 0] + R1, A[a, 1] + R2))

                    # Draw velocities or forces
                    if vectors:
                        line((X[a, 0], X[a, 1], V[a, 0], V[a, 1]), width=2)
            else:
                # Draw unit cell and/or bonds:
                a -= n
                if a < ncell:
                    line((X1[a, 0] + disp[0], X1[a, 1] + disp[1],
                          X2[a, 0] + disp[0], X2[a, 1] + disp[1]))
                else:
                    line((X1[a, 0] + disp[0], X1[a, 1] + disp[1],
                          X2[a, 0] + disp[0], X2[a, 1] + disp[1]), width=bw)

        if self.window['toggle-show-axes']:
            self.draw_axes()

        if self.images.nimages > 1:
            self.draw_frame_number()

        self.window.update()

        if status:
            self.status()

    def draw_axes(self):
        axes_length = 15

        rgb = ['red', 'green', 'blue']

        for i in self.axes[:, 2].argsort():
            a = 20
            b = self.window.size[1] - 20
            c = int(self.axes[i][0] * axes_length + a)
            d = int(-self.axes[i][1] * axes_length + b)
            self.window.line((a, b, c, d))
            self.window.text(c, d, 'XYZ'[i], color=rgb[i])

    def draw_frame_number(self):
        x, y = self.window.size
        self.window.text(x, y, '{0}/{1}'.format(self.frame,
                                                self.images.nimages),
                         anchor='SE')

    def release(self, event):
        if event.button in [4, 5]:
            self.scroll_event(event)
            return

        if event.button != 1:
            return

        selected = self.images.selected
        selected_ordered = self.images.selected_ordered

        if event.time < self.t0 + 200:  # 200 ms
            d = self.P - self.xy
            hit = np.less((d**2).sum(1), (self.scale * self.images.r)**2)
            for a in self.indices[::-1]:
                if a < self.images.natoms and hit[a]:
                    if event.modifier == 'ctrl':
                        selected[a] = not selected[a]
                        if selected[a]:
                            selected_ordered += [a]
                        elif len(selected_ordered) > 0:
                            if selected_ordered[-1] == a:
                                selected_ordered = selected_ordered[:-1]
                            else:
                                selected_ordered = []
                    else:
                        selected[:] = False
                        selected[a] = True
                        selected_ordered = [a]
                    break
            else:
                selected[:] = False
                selected_ordered = []
            self.draw()
        else:
            A = (event.x, event.y)
            C1 = np.minimum(A, self.xy)
            C2 = np.maximum(A, self.xy)
            hit = np.logical_and(self.P > C1, self.P < C2)
            indices = np.compress(hit.prod(1), np.arange(len(hit)))
            if event.modifier != 'ctrl':
                selected[:] = False
            selected[indices] = True
            if (len(indices) == 1 and
                indices[0] not in self.images.selected_ordered):
                selected_ordered += [indices[0]]
            elif len(indices) > 1:
                selected_ordered = []
            self.draw()

        indices = np.arange(self.images.natoms)[self.images.selected]
        if len(indices) != len(selected_ordered):
            selected_ordered = []
        self.images.selected_ordered = selected_ordered

    def press(self, event):
        self.button = event.button
        self.xy = (event.x, event.y)
        self.t0 = event.time
        self.axes0 = self.axes
        self.center0 = self.center

    def move(self, event):
        x = event.x
        y = event.y
        x0, y0 = self.xy
        if self.button == 1:
            x0 = int(round(x0))
            y0 = int(round(y0))
            self.draw()
            self.window.canvas.create_rectangle((x, y, x0, y0))
            return

        if event.modifier == 'shift':
            self.center = (self.center0 -
                           np.dot(self.axes, (x - x0, y0 - y, 0)) / self.scale)
        else:
            # Snap mode: the a-b angle and t should multipla of 15 degrees ???
            a = x - x0
            b = y0 - y
            t = sqrt(a * a + b * b)
            if t > 0:
                a /= t
                b /= t
            else:
                a = 1.0
                b = 0.0
            c = cos(0.01 * t)
            s = -sin(0.01 * t)
            rotation = np.array([(c * a * a + b * b, (c - 1) * b * a, s * a),
                                 ((c - 1) * a * b, c * b * b + a * a, s * b),
                                 (-s * a, -s * b, c)])
            self.axes = np.dot(self.axes0, rotation)
            if self.images.natoms > 0:
                com = self.X[:self.images.natoms].mean(0)
            else:
                com = self.images.A[self.frame].mean(0)
            self.center = com - np.dot(com - self.center0,
                                       np.dot(self.axes0, self.axes.T))
        self.draw(status=False)

    def render_window(self, action):
        Render(self)

    def show_vectors(self, vectors):
        self.vectors = vectors

    def resize(self, event):
        w, h = self.window.size
        self.scale *= (event.width * event.height / (w * h))**0.5
        self.window.size[:] = [event.width, event.height]
        self.draw()
