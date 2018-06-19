from __future__ import print_function, unicode_literals
from ase.gui.i18n import _
import ase.gui.ui as ui
from ase.io.pov import write_pov
from ase.gui.status import formula
from os.path import basename
from os import system
import numpy as np

pack = error = Help = 42


class Render:
    finish_list = [
        'ase2', 'ase3', 'glass', 'simple', 'pale', 'intermediate', 'vmd',
        'jmol'
    ]
    cameras = ['orthographic', 'perspective', 'ultra_wide_angle']
    selection_info_txt = _("""\
    Textures can be used to highlight different parts of
    an atomic structure. This window applies the default
    texture to the entire structure and optionally
    applies a different texture to subsets of atoms that
    can be selected using the mouse.
    An alternative selection method is based on a boolean
    expression in the entry box provided, using the
    variables x, y, z, or Z. For example, the expression
    Z == 11 and x > 10 and y > 10
    will mark all sodium atoms with x or coordinates
    larger than 10. In either case, the button labeled
    `Create new texture from selection` will enable
    to change the attributes of the current selection.
    """)

    def __init__(self, gui):
        self.gui = gui
        ui.Window.__init__(self)
        self.set_title(_('Render current view in povray ... '))
        vbox = ui.VBox()
        vbox.set_border_width(5)
        self.natoms = self.gui.images.natoms
        pack(vbox, [ui.Label(_("Rendering %d atoms.") % self.natoms)])
        self.size = [
            ui.Adjustment(self.gui.width, 1, 9999, 50),
            ui.Adjustment(self.gui.height, 1, 9999, 50)
        ]
        self.width = ui.SpinButton(self.size[0], 0, 0)
        self.height = ui.SpinButton(self.size[1], 0, 0)
        self.render_constraints = ui.CheckButton(_("Render constraints"))
        self.render_constraints.set_sensitive(not self.gui.images.dynamic.all(
        ))
        self.render_constraints.connect("toggled", self.toggle_render_lines)
        pack(vbox, [
            ui.Label(_("Width")), self.width, ui.Label(_("     Height")),
            self.height, ui.Label("       "), self.render_constraints
        ])
        self.width.connect('value-changed', self.change_width, "")
        self.height.connect('value-changed', self.change_height, "")
        self.sizeratio = gui.width / float(gui.height)
        self.line_width = ui.SpinButton(
            ui.Adjustment(0.07, 0.01, 9.99, 0.01), 0, 0)
        self.line_width.set_digits(3)
        self.render_cell = ui.CheckButton(_("Render unit cell"))
        if self.gui.ui.get_widget('/MenuBar/ViewMenu/ShowUnitCell').get_active(
        ):
            self.render_cell.set_active(True)
        else:
            self.render_cell.set_active(False)
            self.render_cell.set_sensitive(False)
        self.render_cell.connect("toggled", self.toggle_render_lines)
        have_lines = (
            not self.gui.images.dynamic.all()) or self.render_cell.get_active()
        self.line_width.set_sensitive(have_lines)
        pack(vbox, [
            ui.Label(_("Line width")), self.line_width,
            ui.Label(_("Angstrom           ")), self.render_cell
        ])
        pack(vbox, [ui.Label("")])
        filename = gui.window.get_title()
        len_suffix = len(filename.split('.')[-1]) + 1
        if len(filename) > len_suffix:
            filename = filename[:-len_suffix]
        self.basename = ui.Entry(max=30)
        self.basename.connect("activate", self.set_outputname, "")
        self.basename.set_text(basename(filename))
        set_name = ui.Button(_("Set"))
        set_name.connect("clicked", self.set_outputname, "")
        pack(vbox, [ui.Label(_("Output basename: ")), self.basename, set_name])
        self.outputname = ui.Label("")
        pack(vbox, [ui.Label(_("               Filename: ")), self.outputname])
        pack(vbox, [ui.Label("")])
        self.tbox = ui.VBox()
        self.tbox.set_border_width(10)
        self.default_texture = ui.combo_box_new_text()
        for t in self.finish_list:
            self.default_texture.append_text(t)
        self.default_texture.set_active(1)
        self.default_transparency = ui.Adjustment(0, 0.0, 1.0, 0.01)
        self.transparency = ui.SpinButton(self.default_transparency, 0, 0)
        self.transparency.set_digits(2)
        pack(self.tbox, [
            ui.Label(_(" Default texture for atoms: ")), self.default_texture,
            ui.Label(_("    transparency: ")), self.transparency
        ])
        pack(self.tbox,
             [ui.Label(_("Define atom selection for new texture:"))])
        self.texture_selection = ui.Entry(max=50)
        self.texture_select_but = ui.Button(_("Select"))
        self.texture_selection.connect("activate", self.select_texture, "")
        self.texture_select_but.connect("clicked", self.select_texture, "")
        pack(self.tbox, [self.texture_selection, self.texture_select_but])
        self.create_texture = ui.Button(_("Create new texture from selection"))
        self.create_texture.connect("clicked", self.material_from_selection,
                                    "")
        self.selection_help_but = ui.Button(_("Help on textures"))
        self.selection_help_but.connect("clicked", self.selection_help, "")
        self.materials = []
        pack(self.tbox, [
            self.create_texture, ui.Label("       "), self.selection_help_but
        ])
        pack(vbox, [self.tbox])
        pack(vbox, [ui.Label("")])
        self.camera_style = ui.combo_box_new_text()
        for c in self.cameras:
            self.camera_style.append_text(c)
        self.camera_style.set_active(0)
        self.camera_distance = ui.SpinButton(
            ui.Adjustment(50.0, -99.0, 99.0, 1.0), 0, 0)
        self.camera_distance.set_digits(1)
        pack(vbox, [
            ui.Label(_("Camera type: ")), self.camera_style,
            ui.Label(_("     Camera distance")), self.camera_distance
        ])
        self.single_frame = ui.RadioButton(None, _("Render current frame"))
        self.nimages = self.gui.images.nimages
        self.iframe = self.gui.frame
        self.movie = ui.RadioButton(self.single_frame,
                                    _("Render all %d frames") % self.nimages)
        self.movie.connect("toggled", self.set_movie)
        self.movie.set_sensitive(self.nimages > 1)
        self.set_outputname()
        pack(vbox, [self.single_frame, ui.Label("   "), self.movie])
        self.transparent = ui.CheckButton(_("Transparent background"))
        self.transparent.set_active(True)
        pack(vbox, [self.transparent])
        self.run_povray = ui.CheckButton(_("Run povray       "))
        self.run_povray.set_active(True)
        self.run_povray.connect("toggled", self.toggle_run_povray, "")
        self.keep_files = ui.CheckButton(_("Keep povray files       "))
        self.keep_files.set_active(False)
        self.keep_files_status = True
        self.window_open = ui.CheckButton(_("Show output window"))
        self.window_open.set_active(True)
        self.window_open_status = True
        pack(vbox, [self.run_povray, self.keep_files, self.window_open])
        pack(vbox, [ui.Label("")])
        cancel_but = ui.Button('Cancel')
        cancel_but.connect('clicked', lambda widget: self.destroy())
        ok_but = ui.Button('OK')
        ok_but.connect('clicked', self.ok)
        close_but = ui.Button('Close')
        close_but.connect('clicked', lambda widget: self.destroy())
        butbox = ui.HButtonBox()
        butbox.pack_start(cancel_but, 0, 0)
        butbox.pack_start(ok_but, 0, 0)
        butbox.pack_start(close_but, 0, 0)
        butbox.show_all()
        pack(vbox, [butbox], end=True, bottom=True)
        self.add(vbox)
        vbox.show()
        self.show()

    def change_width(self, *args):
        self.height.set_value(self.width.get_value() * self.gui.height /
                              float(self.gui.width))

    def change_height(self, *args):
        self.width.set_value(self.height.get_value() * self.gui.width /
                             float(self.gui.height))

    def toggle_render_lines(self, *args):
        self.line_width.set_sensitive(self.render_cell.get_active() or
                                      self.render_constraints.get_active())

    def set_outputname(self, *args):
        movie_index = ''
        self.iframe = self.gui.frame
        if self.movie.get_active():
            while len(movie_index) + len(str(self.iframe)) < len(
                    str(self.nimages)):
                movie_index += '0'
            movie_index = '.' + movie_index + str(self.iframe)
        name = self.basename.get_text() + movie_index + '.pov'
        self.outputname.set_text(name)

    def get_selection(self):
        selection = np.zeros(self.natoms, bool)
        text = self.texture_selection.get_text() or 'False'
        code = compile(text, 'render.py', 'eval')
        for n in range(self.natoms):
            Z = self.gui.images.Z[n]
            x, y, z = self.gui.images.P[self.iframe][n]
            dct = {'n': n, 'Z': Z, 'x': x, 'y': y, 'z': z}
            selection[n] = eval(code, dct)
        return selection

    def select_texture(self, *args):
        self.iframe = self.gui.frame
        self.gui.images.selected = self.get_selection()
        self.gui.set_frame(self.iframe)

    def material_from_selection(self, *args):
        box_selection = self.get_selection()
        selection = self.gui.images.selected.copy()
        if selection.any():
            Z = []
            for n in range(len(selection)):
                if selection[n]:
                    Z += [self.gui.images.Z[n]]
            name = formula(Z)
            if (box_selection == selection).all():
                name += ': ' + self.texture_selection.get_text()
            texture_button = ui.combo_box_new_text()
            for t in self.finish_list:
                texture_button.append_text(t)
            texture_button.set_active(1)
            transparency = ui.Adjustment(0, 0.0, 1.0, 0.01)
            transparency_spin = ui.SpinButton(transparency, 0, 0)
            transparency_spin.set_digits(2)
            delete_button = ui.Button('Delete')
            index = len(self.materials)
            delete_button.connect("clicked", self.delete_material,
                                  {"n": index})
            self.materials += [[
                True, selection, texture_button,
                ui.Label(_("  transparency: ")), transparency_spin,
                ui.Label("   "), delete_button, ui.Label()
            ]]
            self.materials[-1][-1].set_markup("    " + name)
            pack(self.tbox, [
                self.materials[-1][2], self.materials[-1][3],
                self.materials[-1][4], self.materials[-1][5],
                self.materials[-1][6], self.materials[-1][7]
            ])
        else:
            error(_('Can not create new texture! Must have some atoms selected '
                   'to create a new material!'))

    def delete_material(self, button, index, *args):
        n = index["n"]
        self.materials[n][0] = False
        self.materials[n][1] = np.zeros(self.natoms, bool)
        self.materials[n][2].destroy()
        self.materials[n][3].destroy()
        self.materials[n][4].destroy()
        self.materials[n][5].destroy()
        self.materials[n][6].destroy()
        self.materials[n][7].destroy()

    def set_movie(self, *args):
        if self.single_frame.get_active() and self.run_povray.get_active():
            self.window_open.set_active(self.window_open_status)
            self.window_open.set_sensitive(True)
        else:
            if self.run_povray.get_active():
                self.window_open_status = self.window_open.get_active()
            self.window_open.set_active(False)
            self.window_open.set_sensitive(False)
        self.set_outputname()

    def toggle_run_povray(self, *args):
        if self.run_povray.get_active():
            self.keep_files.set_active(self.keep_files_status)
            self.keep_files.set_sensitive(True)
            if self.single_frame.get_active():
                self.window_open.set_active(self.window_open_status)
                self.window_open.set_sensitive(True)
        else:
            self.keep_files_status = self.keep_files.get_active()
            self.keep_files.set_active(True)
            self.keep_files.set_sensitive(False)
            if self.single_frame.get_active():
                self.window_open_status = self.window_open.get_active()
            self.window_open.set_active(False)
            self.window_open.set_sensitive(False)

    def selection_help(self, *args):
        Help(self.selection_info_txt)

    def set_textures(self):
        textures = self.natoms * [
            self.finish_list[self.default_texture.get_active()]
        ]
        for mat in self.materials:
            sel = mat[1]
            t = self.finish_list[mat[2].get_active()]
            if mat[0]:
                for n, val in enumerate(sel):
                    if val:
                        textures[n] = t
        return textures

    def get_colors(self):
        colors = self.gui.get_colors(rgb=True)
        colors_tmp = []
        for n in range(self.natoms):
            colors_tmp.append(
                list(colors[n]) + [0, self.default_transparency.value])
        colors = colors_tmp
        for mat in self.materials:
            sel = mat[1]
            trans = mat[4].get_value()
            for n, val in enumerate(sel):
                if val:
                    colors[n][4] = trans
        return colors

    def ok(self, *args):
        print("Rendering povray image(s): ")
        scale = self.gui.scale * self.height.get_value() / self.gui.height
        bbox = np.empty(4)
        size = np.array(
            [self.width.get_value(), self.height.get_value()]) / scale
        bbox[0:2] = np.dot(self.gui.center, self.gui.axes[:, :2]) - size / 2
        bbox[2:] = bbox[:2] + size
        povray_settings = {
            'run_povray': self.run_povray.get_active(),
            'bbox': bbox,
            'rotation': self.gui.axes,
            'show_unit_cell': self.render_cell.get_active(),
            'display': self.window_open.get_active(),
            'transparent': self.transparent.get_active(),
            'camera_type': self.cameras[self.camera_style.get_active()],
            'camera_dist': self.camera_distance.get_value(),
            'canvas_width': self.width.get_value(),
            'celllinewidth': self.line_width.get_value(),
            'textures': self.set_textures(),
            'exportconstraints': self.render_constraints.get_active()
        }
        if self.single_frame.get_active():
            frames = [self.gui.frame]
        else:
            frames = range(self.nimages)
        initial_frame = self.gui.frame
        for frame in frames:
            self.gui.set_frame(frame)
            povray_settings['colors'] = self.get_colors()
            atoms = self.gui.images.get_atoms(frame)
            self.set_outputname()
            filename = self.outputname.get_text()
            print(" | Writing files for image", filename, "...")
            write_pov(
                filename, atoms, radii=self.gui.images.r, **povray_settings)
            if not self.keep_files.get_active():
                print(" | Deleting temporary file ", filename)
                system("rm " + filename)
                filename = filename[:-4] + '.ini'
                print(" | Deleting temporary file ", filename)
                system("rm " + filename)
        self.gui.set_frame(initial_frame)
        self.set_outputname()
