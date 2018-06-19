from __future__ import unicode_literals

import os
import pickle
import subprocess
import sys
import tempfile
import weakref
from functools import partial
from ase.gui.i18n import _

import numpy as np

from ase import __version__, Atoms
import ase.gui.ui as ui
from ase.gui.calculator import SetCalculator
from ase.gui.crystal import SetupBulkCrystal
from ase.gui.defaults import read_defaults
from ase.gui.energyforces import EnergyForces
from ase.gui.graphene import SetupGraphene
from ase.gui.images import Images
from ase.gui.minimize import Minimize
from ase.gui.nanoparticle import SetupNanoparticle
from ase.gui.nanotube import SetupNanotube
from ase.gui.save import save_dialog
from ase.gui.settings import Settings
from ase.gui.status import Status
from ase.gui.surfaceslab import SetupSurfaceSlab
from ase.gui.view import View


class GUI(View, Status):
    def __init__(self, images=None,
                 rotations='',
                 show_unit_cell=True,
                 show_bonds=False):

        # Try to change into directory of file you are viewing
        try:
            os.chdir(os.path.split(sys.argv[1])[0])
        # This will fail sometimes (e.g. for starting a new session)
        except:
            pass

        if not images:
            images = Images()
            images.initialize([Atoms()])

        self.images = images

        self.config = read_defaults()

        menu = self.get_menu_data(show_unit_cell, show_bonds)

        self.window = ui.ASEGUIWindow(self.exit, menu, self.config,
                                      self.scroll,
                                      self.scroll_event,
                                      self.press, self.move, self.release,
                                      self.resize)

        View.__init__(self, rotations)
        Status.__init__(self)

        self.graphs = []  # list of matplotlib processes
        self.graph_wref = []  # list of weakrefs to Graph objects
        self.movie_window = None
        self.vulnerable_windows = []
        self.simulation = {}  # Used by modules on Calculate menu.
        self.module_state = {}  # Used by modules to store their state.
        self.moving = False

    def run(self, expr=None, test=None):
        self.set_colors()
        self.set_coordinates(self.images.nimages - 1, focus=True)

        if self.images.nimages > 1:
            self.movie()

        if expr is None:
            expr = self.config['gui_graphs_string']

        if expr is not None and expr != '' and self.images.nimages > 1:
            self.plot_graphs(expr=expr)

        if test:
            self.window.test(test)
        else:
            self.window.run()

    def toggle_move_mode(self, key=None):
        self.moving ^= True
        self.draw()

    def step(self, key):
        d = {'Home': -10000000,
             'Page-Up': -1,
             'Page-Down': 1,
             'End': 10000000}[key]
        i = max(0, min(self.images.nimages - 1, self.frame + d))
        self.set_frame(i)
        if self.movie_window is not None:
            self.movie_window.frame_number.value = i

    def _do_zoom(self, x):
        """Utility method for zooming"""
        self.scale *= x
        self.draw()

    def zoom(self, key):
        """Zoom in/out on keypress or clicking menu item"""
        x = {'+': 1.2, '-': 1 / 1.2}[key]
        self._do_zoom(x)

    def scroll_event(self, event):
        """Zoom in/out when using mouse wheel"""
        SHIFT = event.modifier == 'shift'
        x = 1.0
        if event.button == 4:
            x = 1.0 + (1 - SHIFT) * 0.2 + SHIFT * 0.01
        elif event.button == 5:
            x = 1.0 / (1.0 + (1 - SHIFT) * 0.2 + SHIFT * 0.01)
        self._do_zoom(x)

    def settings(self):
        return Settings(self)

    def scroll(self, event):
        CTRL = event.modifier == 'ctrl'
        dxdydz = {'up': (0, 1 - CTRL, CTRL),
                  'down': (0, -1 + CTRL, -CTRL),
                  'right': (1, 0, 0),
                  'left': (-1, 0, 0)}.get(event.key, None)

        if dxdydz is None:
            return

        vec = 0.1 * np.dot(self.axes, dxdydz)
        if event.modifier == 'shift':
            vec *= 0.1

        if self.moving:
            self.images.P[:, self.images.selected] += vec
            self.set_frame()
        else:
            self.center -= vec
            # dx * 0.1 * self.axes[:, 0] - dy * 0.1 * self.axes[:, 1])

            self.draw()

    def delete_selected_atoms(self, widget=None, data=None):
        import ase.gui.ui as ui
        nselected = sum(self.images.selected)
        if nselected and ui.ask_question('Delete atoms',
                                         'Delete selected atoms?'):
            atoms = self.images.get_atoms(self.frame)
            lena = len(atoms)
            for i in range(len(atoms)):
                li = lena - 1 - i
                if self.images.selected[li]:
                    del atoms[li]
            self.new_atoms(atoms)

            self.draw()

    def execute(self):
        from ase.gui.execute import Execute
        Execute(self)

    def constraints_window(self):
        from ase.gui.constraints import Constraints
        Constraints(self)

    def select_all(self, key=None):
        self.images.selected[:] = True
        self.draw()

    def invert_selection(self, key=None):
        self.images.selected[:] = ~self.images.selected
        self.draw()

    def select_constrained_atoms(self, key=None):
        self.images.selected[:] = ~self.images.dynamic
        self.draw()

    def select_immobile_atoms(self, key=None):
        if self.images.nimages > 1:
            R0 = self.images.P[0]
            for R in self.images.P[1:]:
                self.images.selected[:] = ~(np.abs(R - R0) > 1.0e-10).any(1)
        self.draw()

    def movie(self):
        from ase.gui.movie import Movie
        self.movie_window = Movie(self)

    def plot_graphs(self, x=None, expr=None):
        from ase.gui.graphs import Graphs
        g = Graphs(self)
        if expr is not None:
            g.plot(expr=expr)
        self.graph_wref.append(weakref.ref(g))

    def plot_graphs_newatoms(self):
        "Notify any Graph objects that they should make new plots."
        new_wref = []
        found = 0
        for wref in self.graph_wref:
            ref = wref()
            if ref is not None:
                ref.plot()
                new_wref.append(wref)  # Preserve weakrefs that still work.
                found += 1
        self.graph_wref = new_wref
        return found

    def neb(self):
        if len(self.images) <= 1:
            return
        N = self.images.repeat.prod()
        natoms = self.images.natoms // N
        R = self.images.P[:, :natoms]
        E = self.images.E
        F = self.images.F[:, :natoms]
        A = self.images.A[0]
        pbc = self.images.pbc
        process = subprocess.Popen([sys.executable, '-m', 'ase.neb'],
                                   stdin=subprocess.PIPE)
        pickle.dump((E, F, R, A, pbc), process.stdin, protocol=0)
        process.stdin.close()
        self.graphs.append(process)

    def bulk_modulus(self):
        process = subprocess.Popen([sys.executable, '-m', 'ase.eos',
                                    '--plot', '-'],
                                   stdin=subprocess.PIPE)
        v = [abs(np.linalg.det(A)) for A in self.images.A]
        e = self.images.E
        pickle.dump((v, e), process.stdin, protocol=0)
        process.stdin.close()
        self.graphs.append(process)

    def open(self, button=None, filename=None):
        from ase.io.formats import all_formats, get_ioformat

        labels = [_('Automatic')]
        values = ['']

        def key(item):
            return item[1][0]

        for format, (description, code) in sorted(all_formats.items(),
                                                  key=key):
            io = get_ioformat(format)
            if io.read and description != '?':
                labels.append(_(description))
                values.append(format)

        format = [None]

        def callback(value):
            format[0] = value

        chooser = ui.LoadFileDialog(self.window.win, _('Open ...'))
        ui.Label(_('Choose parser:')).pack(chooser.top)
        formats = ui.ComboBox(labels, values, callback)
        formats.pack(chooser.top)

        filename = filename or chooser.go()
        if filename:
            self.images.read([filename], slice(None), format[0])
            self.set_colors()
            self.set_coordinates(self.images.nimages - 1, focus=True)

    def modify_atoms(self, key=None):
        from ase.gui.modify import ModifyAtoms
        ModifyAtoms(self)

    def add_atoms(self, key=None):
        from ase.gui.add import AddAtoms
        AddAtoms(self)

    def quick_info_window(self):
        from ase.gui.quickinfo import info
        ui.Window('Quick Info').add(info(self))

    def bulk_window(self):
        SetupBulkCrystal(self)

    def surface_window(self, menuitem):
        SetupSurfaceSlab(self)

    def nanoparticle_window(self):
        return SetupNanoparticle(self)

    def graphene_window(self, menuitem):
        SetupGraphene(self)

    def nanotube_window(self):
        return SetupNanotube(self)

    def calculator_window(self, menuitem):
        SetCalculator(self)

    def energy_window(self, menuitem):
        EnergyForces(self)

    def energy_minimize_window(self, menuitem):
        Minimize(self)

    def new_atoms(self, atoms, init_magmom=False):
        "Set a new atoms object."
        rpt = getattr(self.images, 'repeat', None)
        self.images.repeat_images(np.ones(3, int))
        self.images.initialize([atoms], init_magmom=init_magmom)
        self.frame = 0  # Prevent crashes
        self.images.repeat_images(rpt)
        self.set_colors()
        self.set_coordinates(frame=0, focus=True)
        self.notify_vulnerable()

    def prepare_new_atoms(self):
        "Marks that the next call to append_atoms should clear the images."
        self.images.prepare_new_atoms()

    def append_atoms(self, atoms):
        "Set a new atoms object."
        # self.notify_vulnerable()   # Do this manually after last frame.
        frame = self.images.append_atoms(atoms)
        self.set_coordinates(frame=frame - 1, focus=True)

    def notify_vulnerable(self):
        """Notify windows that would break when new_atoms is called.

        The notified windows may adapt to the new atoms.  If that is not
        possible, they should delete themselves.
        """
        new_vul = []  # Keep weakrefs to objects that still exist.
        for wref in self.vulnerable_windows:
            ref = wref()
            if ref is not None:
                new_vul.append(wref)
                ref.notify_atoms_changed()
        self.vulnerable_windows = new_vul

    def register_vulnerable(self, obj):
        """Register windows that are vulnerable to changing the images.

        Some windows will break if the atoms (and in particular the
        number of images) are changed.  They can register themselves
        and be closed when that happens.
        """
        self.vulnerable_windows.append(weakref.ref(obj))

    def exit(self, event=None):
        for process in self.graphs:
            process.terminate()
        self.window.close()

    def new(self):
        os.system('ase-gui &')

    def save(self, key=None):
        return save_dialog(self)

    def external_viewer(self, name):
        command = {'xmakemol': 'xmakemol -f',
                   'rasmol': 'rasmol -xyz'}.get(name, name)
        fd, filename = tempfile.mkstemp('.xyz', 'ase.gui-')
        os.close(fd)
        self.images.write(filename)
        os.system('(%s %s &); (sleep 60; rm %s) &' %
                  (command, filename, filename))

    def get_menu_data(self, show_unit_cell, show_bonds):
        M = ui.MenuItem
        return [
            (_('_File'),
             [M(_('_Open'), self.open, 'Ctrl+O'),
              M(_('_New'), self.new, 'Ctrl+N'),
              M(_('_Save'), self.save, 'Ctrl+S'),
              M('---'),
              M(_('_Quit'), self.exit, 'Ctrl+Q')]),

            (_('_Edit'),
             [M(_('Select _all'), self.select_all),
              M(_('_Invert selection'), self.invert_selection),
              M(_('Select _constrained atoms'), self.select_constrained_atoms),
              M(_('Select _immobile atoms'), self.select_immobile_atoms,
                key='Ctrl+I'),
              M('---'),
              # M(_('_Copy'), self.copy_atoms, 'Ctrl+C'),
              # M(_('_Paste'), self.paste_atoms, 'Ctrl+V'),
              M('---'),
              M(_('Hide selected atoms'), self.hide_selected),
              M(_('Show selected atoms'), self.show_selected),
              M('---'),
              M(_('_Modify'), self.modify_atoms, 'Ctrl+Y'),
              M(_('_Add atoms'), self.add_atoms, 'Ctrl+A'),
              M(_('_Delete selected atoms'), self.delete_selected_atoms,
                'Backspace'),
              M('---'),
              M(_('_First image'), self.step, 'Home'),
              M(_('_Previous image'), self.step, 'Page-Up'),
              M(_('_Next image'), self.step, 'Page-Down'),
              M(_('_Last image'), self.step, 'End')]),

            (_('_View'),
             [M(_('Show _unit cell'), self.toggle_show_unit_cell, 'Ctrl+U',
                value=show_unit_cell > 0),
              M(_('Show _axes'), self.toggle_show_axes, value=True),
              M(_('Show _bonds'), self.toggle_show_bonds, 'Ctrl+B',
                value=show_bonds),
              M(_('Show _velocities'), self.toggle_show_velocities, 'Ctrl+G',
                value=False),
              M(_('Show _forces'), self.toggle_show_forces, 'Ctrl+F',
                value=False),
              M(_('Show _Labels'), self.show_labels,
                choices=[_('_None'),
                         _('Atom _Index'),
                         _('_Magnetic Moments'),
                         _('_Element Symbol')]),
              M('---'),
              M(_('Quick Info ...'), self.quick_info_window),
              M(_('Repeat ...'), self.repeat_window, 'R'),
              M(_('Rotate ...'), self.rotate_window),
              M(_('Colors ...'), self.colors_window, 'C'),
              # TRANSLATORS: verb
              M(_('Focus'), self.focus, 'F'),
              M(_('Zoom in'), self.zoom, '+'),
              M(_('Zoom out'), self.zoom, '-'),
              M(_('Change View'),
                submenu=[
                    M(_('Reset View'), self.reset_view, '='),
                    M(_('xy-plane'), self.set_view, 'Z'),
                    M(_('yz-plane'), self.set_view, 'X'),
                    M(_('zx-plane'), self.set_view, 'Y'),
                    M(_('yx-plane'), self.set_view, 'Alt+Z'),
                    M(_('zy-plane'), self.set_view, 'Alt+X'),
                    M(_('xz-plane'), self.set_view, 'Alt+Y'),
                    M(_('a2,a3-plane'), self.set_view, '1'),
                    M(_('a3,a1-plane'), self.set_view, '2'),
                    M(_('a1,a2-plane'), self.set_view, '3'),
                    M(_('a3,a2-plane'), self.set_view, 'Alt+1'),
                    M(_('a1,a3-plane'), self.set_view, 'Alt+2'),
                    M(_('a2,a1-plane'), self.set_view, 'Alt+3')]),
              M(_('Settings ...'), self.settings),
              M('---'),
              M(_('VMD'), partial(self.external_viewer, 'vmd')),
              M(_('RasMol'), partial(self.external_viewer, 'rasmol')),
              M(_('xmakemol'), partial(self.external_viewer, 'xmakemol')),
              M(_('avogadro'), partial(self.external_viewer, 'avogadro'))]),

            (_('_Tools'),
             [M(_('Graphs ...'), self.plot_graphs),
              M(_('Movie ...'), self.movie),
              M(_('Expert mode ...'), self.execute, 'Ctrl+E', disabled=True),
              M(_('Constraints ...'), self.constraints_window),
              M(_('Render scene ...'), self.render_window, disabled=True),
              M(_('_Move atoms'), self.toggle_move_mode, 'Ctrl+M'),
              M(_('NE_B'), self.neb),
              M(_('B_ulk Modulus'), self.bulk_modulus)]),

            # TRANSLATORS: Set up (i.e. build) surfaces, nanoparticles, ...
            (_('_Setup'),
             [M(_('_Bulk Crystal'), self.bulk_window, disabled=True),
              M(_('_Surface slab'), self.surface_window, disabled=True),
              M(_('_Nanoparticle'),
                self.nanoparticle_window),
              M(_('Nano_tube'), self.nanotube_window),
              M(_('Graphene'), self.graphene_window, disabled=True)]),

            (_('_Calculate'),
             [M(_('Set _Calculator'), self.calculator_window, disabled=True),
              M(_('_Energy and Forces'), self.energy_window, disabled=True),
              M(_('Energy Minimization'), self.energy_minimize_window,
                disabled=True)]),

            (_('_Help'),
             [M(_('_About'), partial(ui.about, 'ASE-GUI',
                                     version=__version__,
                                     webpage='https://wiki.fysik.dtu.dk/'
                                     'ase/ase/gui/gui.html')),
              M(_('Webpage ...'), webpage)])]


def webpage():
    import webbrowser
    webbrowser.open('https://wiki.fysik.dtu.dk/ase/ase/gui/gui.html')
