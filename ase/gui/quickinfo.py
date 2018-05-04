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
    if images.natoms < 1:
        txt = _('No atoms loaded.')
    else:
        (nimg, natoms, three) = images.P.shape
        assert three == 3
        img = gui.frame
        uc = images.A[img]
        if nimg > 1:
            equal = True
            for i in range(nimg):
                equal = equal and (uc == images.A[i]).all()
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
                    for periodic in images.pbc]

        # TRANSLATORS: This has the form Periodic: no, no, yes
        pbcstring = _('Periodic: %s, %s, %s') % tuple(periodic)
        txt = format % ((imgtxt, natoms) + tuple(uc.flat) +
                        (pbcstring,) + (uctxt,))
    return txt
