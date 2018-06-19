# -*- coding: utf-8 -*-
"""colors.py - select how to color the atoms in the GUI."""
from __future__ import unicode_literals
from ase.gui.i18n import _

import numpy as np

import ase.gui.ui as ui


class ColorWindow:
    """A window for selecting how to color the atoms."""
    def __init__(self, gui):
        self.win = ui.Window(_('Colors'))
        self.gui = gui
        self.win.add(ui.Label(_('Choose how the atoms are colored:')))
        values = ['jmol', 'tag', 'force', 'velocity', 'charge', 'magmom']
        labels = [_('By atomic number, default "jmol" colors'),
                  _('By tag'),
                  _('By force'),
                  _('By velocity'),
                  _('By charge'),
                  _('By magnetic moment')]

        self.radio = ui.RadioButtons(labels, values, self.toggle,
                                     vertical=True)
        self.radio.value = gui.colormode
        self.win.add(self.radio)
        self.activate()
        self.label = ui.Label()
        self.win.add(self.label)

    def activate(self):
        images = self.gui.images
        radio = self.radio
        radio['tag'].active = images.T.any()
        radio['force'].active = np.isfinite(images.F).all()
        radio['velocity'].active = np.isfinite(images.V).all()
        radio['charge'].active = images.q.any()
        radio['magmom'].active = images.M.any()

    def toggle(self, value):
        if value == 'jmol':
            self.gui.set_colors()
            text = ''
        else:
            self.gui.colormode = value
            scalars = np.array([self.gui.get_color_scalars(i)
                                for i in range(len(self.gui.images))])
            mn = scalars.min()
            mx = scalars.max()
            colorscale = ['#{0:02X}AA00'.format(red)
                          for red in range(0, 240, 10)]
            self.gui.colormode_data = colorscale, mn, mx

            unit = {'tag': '',
                    'force': 'eV/Ang',
                    'velocity': '??',
                    'charge': '|e|',
                    u'magmom': 'μB'}[value]
            text = '[{0},{1}]: [{2:.6f},{3:.6f}] {4}'.format(
                _('Green'), _('Yellow'), mn, mx, unit)

        self.label.text = text
        self.radio.value = value
        self.gui.draw()
        return text  # for testing

    def notify_atoms_changed(self):
        "Called by gui object when the atoms have changed."
        self.activate()
        mode = self.gui.colormode
        if not self.radio[mode].active:
            mode = 'jmol'
        self.toggle(mode)
