from __future__ import unicode_literals
import pickle
import subprocess
import sys

import numpy as np

from ase.gui.i18n import _
import ase.gui.ui as ui

graph_help_text = _("""\
Symbols:
<c>e</c>: total energy
<c>epot</c>: potential energy
<c>ekin</c>: kinetic energy
<c>fmax</c>: maximum force
<c>fave</c>: average force
<c>R[n,0-2]</c>: position of atom number <c>n</c>
<c>d(n<sub>1</sub>,n<sub>2</sub>)</c>: distance between two atoms \
<c>n<sub>1</sub></c> and <c>n<sub>2</sub></c>
<c>i</c>: current image number
<c>E[i]</c>: energy of image number <c>i</c>
<c>F[n,0-2]</c>: force on atom number <c>n</c>
<c>V[n,0-2]</c>: velocity of atom number <c>n</c>
<c>M[n]</c>: magnetic moment of atom number <c>n</c>
<c>A[0-2,0-2]</c>: unit-cell basis vectors
<c>s</c>: path length
<c>a(n1,n2,n3)</c>: angle between atoms <c>n<sub>1</sub></c>, \
<c>n<sub>2</sub></c> and <c>n<sub>3</sub></c>, centered on <c>n<sub>2</sub></c>
<c>dih(n1,n2,n3,n4)</c>: dihedral angle between <c>n<sub>1</sub></c>, \
<c>n<sub>2</sub></c>, <c>n<sub>3</sub></c> and <c>n<sub>4</sub></c>
<c>T</c>: temperature (K)\
""")


class Graphs:
    def __init__(self, gui):
        win = ui.Window('Graphs')
        self.expr = ui.Entry('', 50, self.plot)
        win.add([self.expr, ui.helpbutton(graph_help_text)])

        win.add([ui.Button(_('Plot'), self.plot, 'xy'),
                 ' x, y1, y2, ...'], 'w')
        win.add([ui.Button(_('Plot'), self.plot, 'y'),
                 ' y1, y2, ...'], 'w')
        win.add([ui.Button(_('Save'), self.save),
                 ui.Button(_('Clear'), self.clear)], 'w')

        self.gui = gui

    def plot(self, type=None, expr=None):
        if expr is None:
            expr = self.expr.value
        else:
            self.expr.value = expr

        try:
            data = self.gui.images.graph(expr)
        except (SyntaxError, NameError) as ex:
            ui.error(ex)
            return

        process = subprocess.Popen([sys.executable, '-m', 'ase.gui.graphs'],
                                   stdin=subprocess.PIPE)
        pickledata = (data, self.gui.frame, expr, type)
        pickle.dump(pickledata, process.stdin, protocol=0)
        process.stdin.close()
        self.gui.graphs.append(process)

    def save(self):
        dialog = ui.SaveFileDialog(self.gui.window.win,
                                   _('Save data to file ... '))
        filename = dialog.go()
        if filename:
            expr = self.expr.value
            data = self.gui.images.graph(expr)
            np.savetxt(filename, data.T, header=expr)

    def clear(self):
        import matplotlib.pyplot as plt
        for fig in self.gui.graphs:
            plt.close(fig)
        self.gui.graphs = []


def make_plot(data, i, expr, type):
    import matplotlib.pyplot as plt
    basesize = 4
    plt.figure(figsize=(basesize * 2.5**0.5, basesize))
    m = len(data)

    if type is None:
        if m == 1:
            type = 'y'
        else:
            type = 'xy'

    if type == 'y':
        for j in range(m):
            plt.plot(data[j])
            plt.plot([i], [data[j, i]], 'o')
    else:
        for j in range(1, m):
            plt.plot(data[0], data[j])
            plt.plot([data[0, i]], [data[j, i]], 'o')
    plt.title(expr)
    plt.show()


if __name__ == '__main__':
    if sys.version_info[0] == 2:
        make_plot(*pickle.load(sys.stdin))
    else:
        make_plot(*pickle.load(sys.stdin.buffer))
