"Module for displaying information about the system."

from __future__ import unicode_literals
from ase.gui.i18n import _

singleimage = _('Single image loaded.')
multiimage = _('Image %d loaded (0 - %d).')
ucconst = _('Unit cell is fixed.')
ucvaries = _('Unit cell varies.')

format = _("""\
%s

Number of atoms: %d.

Unit cell:
  %8.3f  %8.3f  %8.3f
  %8.3f  %8.3f  %8.3f
  %8.3f  %8.3f  %8.3f

%s
%s
""")


def info(gui):
    images = gui.images
    nimg = len(images)
    atoms = gui.atoms
    natoms = len(atoms)

    if len(atoms) < 1:
        txt = _('This frame has no atoms.')
    else:
        img = gui.frame

        uc = atoms.cell
        if nimg > 1:
            equal = True
            for i in range(nimg):
                equal = equal and (uc == images[i].cell).all()
            if equal:
                uctxt = ucconst
            else:
                uctxt = ucvaries
        else:
            uctxt = ''
        if nimg == 1:
            imgtxt = singleimage
        else:
            imgtxt = multiimage % (img, nimg - 1)

        periodic = [[_('no'), _('yes')][periodic]
                    for periodic in atoms.pbc]

        # TRANSLATORS: This has the form Periodic: no, no, yes
        pbcstring = _('Periodic: %s, %s, %s') % tuple(periodic)
        txt = format % ((imgtxt, natoms) + tuple(uc.flat) +
                        (pbcstring,) + (uctxt,))
        if atoms.number_of_lattice_vectors == 3:
            txt += _('Volume: ') + '{:8.3f}'.format(atoms.get_volume())
    return txt
