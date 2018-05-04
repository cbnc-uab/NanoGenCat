from ase.io.utils import generate_writer_variables, make_patch_list


class Matplotlib:
    def __init__(self, atoms, ax,
                 rotation='', show_unit_cell=False, radii=None,
                 colors=None, scale=1, offset=(0, 0)):
        generate_writer_variables(
            self, atoms, rotation=rotation,
            show_unit_cell=show_unit_cell,
            radii=radii, bbox=None, colors=colors, scale=scale,
            extra_offset=offset)

        self.ax = ax
        self.figure = ax.figure
        self.ax.set_aspect('equal')

    def write(self):
        self.write_body()
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(0, self.h)

    def write_body(self):
        patch_list = make_patch_list(self)
        for patch in patch_list:
            self.ax.add_patch(patch)


def plot_atoms(atoms, ax=None, **parameters):
    """Plot an atoms object in a matplotlib subplot.

    Parameters
    ----------
    atoms : Atoms object
    ax : Matplotlib subplot object
    rotation : str, optional
        In degrees. In the form '10x,20y,30z'
    show_unit_cell : bool, optional, default False
        Draw the bounds of the atoms object as dashed lines.
    radii : float, optional
        The radii of the atoms
    colors : list of strings, optional
        Color of the atoms, must be the same length as
        the number of atoms in the atoms object.
    scale : float, optional
        Scaling of the plotted atoms and lines.
    offset : tuple (float, float), optional
        Offset of the plotted atoms and lines.
    """
    if isinstance(atoms, list):
        assert len(atoms) == 1
        atoms = atoms[0]

    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    Matplotlib(atoms, ax, **parameters).write()
    return ax
