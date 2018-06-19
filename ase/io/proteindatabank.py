"""Module to read and write atoms in PDB file format.

See::

    http://www.wwpdb.org/documentation/file-format

Note: The PDB format saves cell lengths and angles; hence the absolute
orientation is lost when saving.  Saving and loading a file will
conserve the scaled positions, not the absolute ones.
"""

import warnings

import numpy as np

from ase.atoms import Atom, Atoms
from ase.parallel import paropen
from ase.geometry import cellpar_to_cell
from ase.utils import basestring


def read_proteindatabank(fileobj, index=-1):
    """Read PDB files."""

    if isinstance(fileobj, basestring):
        fileobj = open(fileobj)

    images = []
    orig = np.identity(3)
    trans = np.zeros(3)
    atoms = Atoms()
    for line in fileobj.readlines():
        if line.startswith('CRYST1'):
            cellpar = [float(word) for word in line[6:54].split()]
            atoms.set_cell(cellpar_to_cell(cellpar))
            atoms.pbc = True
        for c in range(3):
            if line.startswith('ORIGX' + '123'[c]):
                pars = [float(word) for word in line[10:55].split()]
                orig[c] = pars[:3]
                trans[c] = pars[3]

        if line.startswith('ATOM') or line.startswith('HETATM'):
            try:
                # Atom name is arbitrary and does not necessarily
                # contain the element symbol.  The specification
                # requires the element symbol to be in columns 77+78.
                symbol = line[76:78].strip().lower().capitalize()
                words = line[30:55].split()
                position = np.array([float(words[0]),
                                     float(words[1]),
                                     float(words[2])])
                position = np.dot(orig, position) + trans
                atoms.append(Atom(symbol, position))
            except Exception as ex:
                warnings.warn('Discarding atom when reading PDB file: {}'
                              .format(ex))
        if line.startswith('ENDMDL'):
            images.append(atoms)
            atoms = Atoms()
    if len(images) == 0:
        images.append(atoms)
    return images[index]


def write_proteindatabank(fileobj, images):
    """Write images to PDB-file."""
    if isinstance(fileobj, basestring):
        fileobj = paropen(fileobj, 'w')

    if hasattr(images, 'get_positions'):
        images = [images]


    rotation = None
    if images[0].get_pbc().any():
        from ase.geometry import cell_to_cellpar, cellpar_to_cell

        currentcell = images[0].get_cell()
        cellpar = cell_to_cellpar(currentcell)
        exportedcell = cellpar_to_cell(cellpar)
        rotation = np.linalg.solve(currentcell, exportedcell)
        # ignoring Z-value, using P1 since we have all atoms defined explicitly
        format = 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1\n'
        fileobj.write(format % (cellpar[0], cellpar[1], cellpar[2],
                                cellpar[3], cellpar[4], cellpar[5]))

    #     1234567 123 6789012345678901   89   67   456789012345678901234567 890
    format = ('ATOM  %5d %4s MOL     1    %8.3f%8.3f%8.3f  1.00  0.00'
              '          %2s  \n')

    # RasMol complains if the atom index exceeds 100000. There might
    # be a limit of 5 digit numbers in this field.
    MAXNUM = 100000

    symbols = images[0].get_chemical_symbols()
    natoms = len(symbols)

    for n, atoms in enumerate(images):
        fileobj.write('MODEL     ' + str(n + 1) + '\n')
        p = atoms.get_positions()
        if rotation is not None:
            p = p.dot(rotation)
        for a in range(natoms):
            x, y, z = p[a]
            fileobj.write(format % (a % MAXNUM, symbols[a],
                                    x, y, z, symbols[a].upper()))
        fileobj.write('ENDMDL\n')
