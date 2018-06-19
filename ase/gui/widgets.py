from __future__ import unicode_literals
from ase.gui.i18n import _

import ase.data
import ase.gui.ui as ui


class Element(list):
    def __init__(self, symbol='', callback=None):
        list.__init__(self,
                      [_('Element:'),
                       ui.Entry(symbol, 3, self.enter),
                       ui.Label('', 'red')])
        self.callback = callback
        self._symbol = None
        self._Z = None

    @property
    def symbol(self):
        self.check()
        return self._symbol

    @symbol.setter
    def symbol(self, value):
        self[1].value = value

    @property
    def Z(self):
        self.check()
        return self._Z

    @Z.setter
    def Z(self, value):
        self.symbol = ase.data.chemical_symbols[value]

    def check(self):
        self._symbol = self[1].value
        if not self._symbol:
            self.error(_('No element specified!'))
            return False
        self._Z = ase.data.atomic_numbers.get(self._symbol)
        if self._Z is None:
            try:
                self._Z = int(self._symbol)
            except ValueError:
                self.error()
                return False
            self._symbol = ase.data.chemical_symbols[self._Z]
        self[2].text = ''
        return True

    def enter(self):
        self.check()
        self.callback(self)

    def error(self, text=_('ERROR: Invalid element!')):
        self._symbol = None
        self._Z = None
        self[2].text = text


def pybutton(title, callback):
    """A button for displaying Python code.

    When pressed, it opens a window displaying some Python code, or an error
    message if no Python code is ready.
    """
    return ui.Button('Python', pywindow, title, callback)


def pywindow(title, callback):
    code = callback()
    if code is None:
        ui.error(
            _('No Python code'),
            _('You have not (yet) specified a consistent set of parameters.'))
    else:
        win = ui.Window(title)
        win.add(ui.Text(code))
