import time

import numpy as np

from ase.constraints import Filter, FixAtoms
from ase.geometry.cell import cell_to_cellpar

from ase.optimize.precon import logger

try:
    from matscipy.neighbours import neighbour_list
    have_matscipy = True
except ImportError:
    from ase.neighborlist import NeighborList
    have_matscipy = False

def get_neighbours(atoms, r_cut, self_interaction=False):
    """Return a list of pairs of atoms within a given distance of each other.

    If matscipy can be imported, then this will directly call matscipy's
    neighbourlist function. Otherwise it will use ASE's NeighborList object.

    Args:
        atoms: ase.atoms object to calculate neighbours for
        r_cut: cutoff radius (float). Pairs of atoms are considered neighbours
            if they are within a distance r_cut of each other (note that this
            is double the parameter used in the ASE's neighborlist module)

    Returns: a tuple (i_list, j_list, d_list, fixed_atoms):
        i_list, j_list: i and j indices of each neighbour pair
        d_list: absolute distance between the corresponding pair
        fixed_atoms: indices of any fixed atoms
    """

    if isinstance(atoms, Filter):
        atoms = atoms.atoms

    if have_matscipy:
        i_list, j_list, d_list = neighbour_list('ijd', atoms, r_cut)
    else:

        radii = [r_cut / 2 for i in range(len(atoms))]
        nl = NeighborList(radii,
                          sorted=False,
                          self_interaction=False,
                          bothways=True)
        nl.update(atoms)
        i_list = []
        j_list = []
        d_list = []

        for i, atom in enumerate(atoms):
            posn_i = atom.position
            indices, offsets = nl.get_neighbors(i)
            assert len(indices) == len(offsets)
            for j, offset in zip(indices, offsets):
                # Offsets represent how far away an atom is from its pair in terms
                # of the repeating cell - for instance, an atom i might be in cell
                # (0, 0, 0) while the neighbouring atom j is in cell (0, 1, 1). To
                # get the true position we have to correct for the offset:
                posn_j = atoms.positions[j] + np.dot(offset, atoms.get_cell())
                distance = np.sqrt(((posn_j - posn_i)**2).sum())
                i_list.append(i)
                j_list.append(j)
                d_list.append(distance)

        i_list = np.array(i_list)
        j_list = np.array(j_list)
        d_list = np.array(d_list)

    # filter out self-interactions (across PBC)
    if not self_interaction:
        mask = i_list != j_list
        i_list = i_list[mask]
        j_list = j_list[mask]
        d_list = d_list[mask]

    # filter out bonds where 1st atom (i) in pair is fixed
    fixed_atoms = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed_atoms.extend(list(constraint.index))
        else:
            raise TypeError(
                'only FixAtoms constraints are supported by Precon class')

    return i_list, j_list, d_list, fixed_atoms


def estimate_nearest_neighbour_distance(atoms):
    """
    Estimate nearest neighbour distance r_NN

    Args:
        atoms: Atoms object

    Returns:
        rNN: float
            Nearest neighbour distance
    """

    if isinstance(atoms, Filter):
        atoms = atoms.atoms

    start_time = time.time()
    # compute number of neighbours of each atom. If any atom doesn't
    # have a neighbour we increase the cutoff and try again, until our
    # cutoff exceeds the size of the sytem
    r_cut = 1.0
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # Golden ratio

    # cell lengths and angles
    a, b, c, alpha, beta, gamma = cell_to_cellpar(atoms.cell)
    extent = [a, b, c]
    logger.debug('estimate_nearest_neighbour_distance(): extent=%r',
                 extent)

    while r_cut < 2.0 * max(extent):
        logger.info('estimate_nearest_neighbour_distance(): '
                    'calling neighbour_list with r_cut=%.2f A', r_cut)
        i, j, rij, fixed_atoms = get_neighbours(
            atoms, r_cut, self_interaction=True)
        if len(i) != 0:
            nn_i = np.bincount(i, minlength=len(atoms))
            if (nn_i != 0).all():
                break
        r_cut *= phi
    else:
        raise RuntimeError('increased r_cut to twice system extent without '
                           'finding neighbours for all atoms. This can '
                           'happen if your system is too small; try '
                           'setting r_cut manually')

    # maximum of nearest neigbour distances
    nn_distances = [np.min(rij[i == I]) for I in range(len(atoms))]
    r_NN = np.max(nn_distances)

    logger.info('estimate_nearest_neighbour_distance(): got r_NN=%.3f in %s s',
                r_NN, time.time() - start_time)
    return r_NN
