from __future__ import unicode_literals
from ase.gui.i18n import _

import ase.gui.ui as ui
from ase.gui.widgets import Element


def txt2pos(txt):
    try:
        x, y, z = (float(x) for x in txt.split(','))
    except ValueError as ex:
        ui.error(_('Bad position'), ex)
    else:
        return x, y, z


class AddAtoms:
    def __init__(self, gui):
        win = ui.Window(_('Add atoms'))
        self.element = Element()
        win.add(self.element)
        self.absolute_position = ui.Entry('0,0,0')
        self.relative_position = ui.Entry('1.5,0,0')
        win.add([_('Absolute position:'),
                 self.absolute_position,
                 ui.Button(_('Add'), self.add_absolute)])
        win.add([_('Relative to average position (of selection):'),
                 self.relative_position,
                 ui.Button(_('Add'), self.add_relative)])
        self.gui = gui

    def add_absolute(self):
        pos = txt2pos(self.absolute_position.value)
        self.add(pos)

    def add_relative(self):
        rpos = txt2pos(self.relative_position.value)
        pos = self.gui.images.P[self.gui.frame]
        if self.gui.images.selected.any():
            pos = pos[self.gui.images.selected]
        center = pos.mean(0)
        self.add(center + rpos)

    def add(self, pos):
        if pos is None or self.element.symbol is None:
            return
        atoms = self.gui.images.get_atoms(self.gui.frame)
        atoms.append(self.element.symbol)
        atoms.positions[-1] = pos
        self.gui.new_atoms(atoms, init_magmom=True)
        self.gui.images.selected[:] = False
        self.gui.images.selected[-1] = True
        self.gui.draw()
