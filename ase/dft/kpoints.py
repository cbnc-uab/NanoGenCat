from __future__ import division
from ase.utils import basestring
import re
import warnings
from math import sin, cos, pi

import numpy as np

from ase.geometry import cell_to_cellpar, crystal_structure_from_cell


def monkhorst_pack(size):
    """Construct a uniform sampling of k-space of given size."""
    if np.less_equal(size, 0).any():
        raise ValueError('Illegal size: %s' % list(size))
    kpts = np.indices(size).transpose((1, 2, 3, 0)).reshape((-1, 3))
    return (kpts + 0.5) / size - 0.5


def get_monkhorst_pack_size_and_offset(kpts):
    """Find Monkhorst-Pack size and offset.

    Returns (size, offset), where::

        kpts = monkhorst_pack(size) + offset.

    The set of k-points must not have been symmetry reduced."""

    if len(kpts) == 1:
        return np.ones(3, int), np.array(kpts[0], dtype=float)

    size = np.zeros(3, int)
    for c in range(3):
        # Determine increment between k-points along current axis
        delta = max(np.diff(np.sort(kpts[:, c])))

        # Determine number of k-points as inverse of distance between kpoints
        if delta > 1e-8:
            size[c] = int(round(1.0 / delta))
        else:
            size[c] = 1

    if size.prod() == len(kpts):
        kpts0 = monkhorst_pack(size)
        offsets = kpts - kpts0

        # All offsets must be identical:
        if (offsets.ptp(axis=0) < 1e-9).all():
            return size, offsets[0].copy()

    raise ValueError('Not an ASE-style Monkhorst-Pack grid!')


def get_monkhorst_shape(kpts):
    warnings.warn('Use get_monkhorst_pack_size_and_offset()[0] instead.')
    return get_monkhorst_pack_size_and_offset(kpts)[0]


def kpoint_convert(cell_cv, skpts_kc=None, ckpts_kv=None):
    """Convert k-points between scaled and cartesian coordinates.

    Given the atomic unit cell, and either the scaled or cartesian k-point
    coordinates, the other is determined.

    The k-point arrays can be either a single point, or a list of points,
    i.e. the dimension k can be empty or multidimensional.
    """
    if ckpts_kv is None:
        icell_cv = 2 * np.pi * np.linalg.pinv(cell_cv).T
        return np.dot(skpts_kc, icell_cv)
    elif skpts_kc is None:
        return np.dot(ckpts_kv, cell_cv.T) / (2 * np.pi)
    else:
        raise KeyError('Either scaled or cartesian coordinates must be given.')


def parse_path_string(s):
    """Parse compact string representation of BZ path.

    A path string can have several non-connected sections separated by
    commas. The return value is a list of sections where each section is a
    list of labels.

    Examples:

    >>> parse_path_string('GX')
    [['G', 'X']]
    >>> parse_path_string('GX,M1A')
    [['G', 'X'], ['M1', 'A']]
    """
    paths = []
    for path in s.split(','):
        names = [name if name != 'Gamma' else 'G'
                 for name in re.split(r'([A-Z][a-z0-9]*)', path)
                 if name]
        paths.append(names)
    return paths


def bandpath(path, cell, npoints=50):
    """Make a list of kpoints defining the path between the given points.

    path: list or str
        Can be:

        * a string that parse_path_string() understands: 'GXL'
        * a list of BZ points: [(0, 0, 0), (0.5, 0, 0)]
        * or several lists of BZ points if the the path is not continuous.
    cell: 3x3
        Unit cell of the atoms.
    npoints: int
        Length of the output kpts list.

    Return list of k-points, list of x-coordinates and list of
    x-coordinates of special points."""

    if isinstance(path, basestring):
        xtal = crystal_structure_from_cell(cell)
        special = get_special_points(xtal, cell)
        paths = []
        for names in parse_path_string(path):
            paths.append([special[name] for name in names])
    elif np.array(path[0]).ndim == 1:
        paths = [path]
    else:
        paths = path

    points = np.concatenate(paths)
    dists = points[1:] - points[:-1]
    lengths = [np.linalg.norm(d) for d in kpoint_convert(cell, skpts_kc=dists)]
    i = 0
    for path in paths[:-1]:
        i += len(path)
        lengths[i - 1] = 0

    length = sum(lengths)
    kpts = []
    x0 = 0
    x = []
    X = [0]
    for P, d, L in zip(points[:-1], dists, lengths):
        n = max(2, int(round(L * (npoints - len(x)) / (length - x0))))
        for t in np.linspace(0, 1, n)[:-1]:
            kpts.append(P + t * d)
            x.append(x0 + t * L)
        x0 += L
        X.append(x0)
    kpts.append(points[-1])
    x.append(x0)
    return np.array(kpts), np.array(x), np.array(X)


get_bandpath = bandpath  # old name


def labels_from_kpts(kpts, cell, crystal_structure=None, eps=1e-5):
    """Get an x-axis to be used when plotting a band structure.

    The first of the returned lists can be used as a x-axis when plotting
    the band structure. The second list can be used as xticks, and the third
    as xticklabels.

    Parameters:

    kpts: list
        List of scaled k-points.

    cell: list
        Unit cell of the atomic structure.

    crystal_structure: str
        Crystal structure of the atoms. If None is provided the crystal
        structure is determined from the cell.

    Returns:

    Three arrays; the first is a list of cumulative distances between kpoints,
    the second is x coordinates of the special points,
    the third is the special points as strings.
     """
    if crystal_structure is None:
        crystal_structure = crystal_structure_from_cell(cell)

    points = np.asarray(kpts)
    diffs = points[1:] - points[:-1]
    kinks = abs(diffs[1:] - diffs[:-1]).sum(1) > eps
    N = len(points)
    indices = [0]
    indices.extend(np.arange(1, N - 1)[kinks])
    indices.append(N - 1)

    special = get_special_points(crystal_structure, cell)
    labels = []
    for kpt in points[indices]:
        for label, k in special.items():
            if abs(kpt - k).sum() < eps:
                break
        else:
            label = '?'
        labels.append(label)

    xcoords = [0]
    for i1, i2 in zip(indices[:-1], indices[1:]):
        if i1 + 1 == i2:
            length = 0
        else:
            diff = points[i2] - points[i1]
            length = np.linalg.norm(kpoint_convert(cell, skpts_kc=diff))
        xcoords.extend(np.linspace(0, length, i2 - i1 + 1)[1:] + xcoords[-1])

    xcoords = np.array(xcoords)
    return xcoords, xcoords[indices], labels


special_points = {
    'cubic': {'G': [0, 0, 0],
              'M': [1 / 2, 1 / 2, 0],
              'R': [1 / 2, 1 / 2, 1 / 2],
              'X': [0, 1 / 2, 0]},
    'fcc': {'G': [0, 0, 0],
            'K': [3 / 8, 3 / 8, 3 / 4],
            'L': [1 / 2, 1 / 2, 1 / 2],
            'U': [5 / 8, 1 / 4, 5 / 8],
            'W': [1 / 2, 1 / 4, 3 / 4],
            'X': [1 / 2, 0, 1 / 2]},
    'bcc': {'G': [0, 0, 0],
            'H': [1 / 2, -1 / 2, 1 / 2],
            'P': [1 / 4, 1 / 4, 1 / 4],
            'N': [0, 0, 1 / 2]},
    'tetragonal': {'G': [0, 0, 0],
                   'A': [1 / 2, 1 / 2, 1 / 2],
                   'M': [1 / 2, 1 / 2, 0],
                   'R': [0, 1 / 2, 1 / 2],
                   'X': [0, 1 / 2, 0],
                   'Z': [0, 0, 1 / 2]},
    'orthorhombic': {'G': [0, 0, 0],
                     'R': [1 / 2, 1 / 2, 1 / 2],
                     'S': [1 / 2, 1 / 2, 0],
                     'T': [0, 1 / 2, 1 / 2],
                     'U': [1 / 2, 0, 1 / 2],
                     'X': [1 / 2, 0, 0],
                     'Y': [0, 1 / 2, 0],
                     'Z': [0, 0, 1 / 2]},
    'hexagonal': {'G': [0, 0, 0],
                  'A': [0, 0, 1 / 2],
                  'H': [1 / 3, 1 / 3, 1 / 2],
                  'K': [1 / 3, 1 / 3, 0],
                  'L': [1 / 2, 0, 1 / 2],
                  'M': [1 / 2, 0, 0]}}


special_paths = {
    'cubic': 'GXMGRX,MR',
    'fcc': 'GXWKGLUWLK,UX',
    'bcc': 'GHNGPH,PN',
    'tetragonal': 'GXMGZRAZXR,MA',
    'orthorhombic': 'GXSYGZURTZ,YT,UX,SR',
    'hexagonal': 'GMKGALHA,LM,KH',
    'monoclinic': 'GYHCEM1AXH1,MDZ,YD'}


def get_special_points(lattice, cell, eps=1e-4):
    """Return dict of special points.

    The definitions are from a paper by Wahyu Setyawana and Stefano
    Curtarolo::

        http://dx.doi.org/10.1016/j.commatsci.2010.05.010

    lattice: str
        One of the following: cubic, fcc, bcc, orthorhombic, tetragonal,
        hexagonal or monoclinic.
    cell: 3x3 ndarray
        Unit cell.
    eps: float
        Tolerance for cell-check.
    """

    lattice = lattice.lower()

    cellpar = cell_to_cellpar(cell=cell)
    abc = cellpar[:3]
    angles = cellpar[3:] / 180 * pi
    a, b, c = abc
    alpha, beta, gamma = angles

    # Check that the unit-cells are as in the Setyawana-Curtarolo paper:
    if lattice == 'cubic':
        assert abc.ptp() < eps and abs(angles - pi / 2).max() < eps
    elif lattice == 'fcc':
        assert abc.ptp() < eps and abs(angles - pi / 3).max() < eps
    elif lattice == 'bcc':
        angle = np.arccos(-1 / 3)
        assert abc.ptp() < eps and abs(angles - angle).max() < eps
    elif lattice == 'tetragonal':
        assert abs(a - b) < eps and abs(angles - pi / 2).max() < eps
    elif lattice == 'orthorhombic':
        assert abs(angles - pi / 2).max() < eps
    elif lattice == 'hexagonal':
        assert abs(a - b) < eps
        assert abs(gamma - pi / 3 * 2) < eps
        assert abs(angles[:2] - pi / 2).max() < eps
    elif lattice == 'monoclinic':
        assert c >= a and c >= b
        assert alpha < pi / 2 and abs(angles[1:] - pi / 2).max() < eps

    if lattice != 'monoclinic':
        return special_points[lattice]

    # Here, we need the cell:
    eta = (1 - b * cos(alpha) / c) / (2 * sin(alpha)**2)
    nu = 1 / 2 - eta * c * cos(alpha) / b
    return {'G': [0, 0, 0],
            'A': [1 / 2, 1 / 2, 0],
            'C': [0, 1 / 2, 1 / 2],
            'D': [1 / 2, 0, 1 / 2],
            'D1': [1 / 2, 0, -1 / 2],
            'E': [1 / 2, 1 / 2, 1 / 2],
            'H': [0, eta, 1 - nu],
            'H1': [0, 1 - eta, nu],
            'H2': [0, eta, -nu],
            'M': [1 / 2, eta, 1 - nu],
            'M1': [1 / 2, 1 - eta, nu],
            'M2': [1 / 2, eta, -nu],
            'X': [0, 1 / 2, 0],
            'Y': [0, 0, 1 / 2],
            'Y1': [0, 0, -1 / 2],
            'Z': [1 / 2, 0, 0]}


# ChadiCohen k point grids. The k point grids are given in units of the
# reciprocal unit cell. The variables are named after the following
# convention: cc+'<Nkpoints>'+_+'shape'. For example an 18 k point
# sq(3)xsq(3) is named 'cc18_sq3xsq3'.

cc6_1x1 = np.array([
    1, 1, 0, 1, 0, 0, 0, -1, 0, -1, -1, 0, -1, 0, 0,
    0, 1, 0]).reshape((6, 3)) / 3.0

cc12_2x3 = np.array([
    3, 4, 0, 3, 10, 0, 6, 8, 0, 3, -2, 0, 6, -4, 0,
    6, 2, 0, -3, 8, 0, -3, 2, 0, -3, -4, 0, -6, 4, 0, -6, -2, 0, -6,
    -8, 0]).reshape((12, 3)) / 18.0

cc18_sq3xsq3 = np.array([
    2, 2, 0, 4, 4, 0, 8, 2, 0, 4, -2, 0, 8, -4,
    0, 10, -2, 0, 10, -8, 0, 8, -10, 0, 2, -10, 0, 4, -8, 0, -2, -8,
    0, 2, -4, 0, -4, -4, 0, -2, -2, 0, -4, 2, 0, -2, 4, 0, -8, 4, 0,
    -4, 8, 0]).reshape((18, 3)) / 18.0

cc18_1x1 = np.array([
    2, 4, 0, 2, 10, 0, 4, 8, 0, 8, 4, 0, 8, 10, 0,
    10, 8, 0, 2, -2, 0, 4, -4, 0, 4, 2, 0, -2, 8, 0, -2, 2, 0, -2, -4,
    0, -4, 4, 0, -4, -2, 0, -4, -8, 0, -8, 2, 0, -8, -4, 0, -10, -2,
    0]).reshape((18, 3)) / 18.0

cc54_sq3xsq3 = np.array([
    4, -10, 0, 6, -10, 0, 0, -8, 0, 2, -8, 0, 6,
    -8, 0, 8, -8, 0, -4, -6, 0, -2, -6, 0, 2, -6, 0, 4, -6, 0, 8, -6,
    0, 10, -6, 0, -6, -4, 0, -2, -4, 0, 0, -4, 0, 4, -4, 0, 6, -4, 0,
    10, -4, 0, -6, -2, 0, -4, -2, 0, 0, -2, 0, 2, -2, 0, 6, -2, 0, 8,
    -2, 0, -8, 0, 0, -4, 0, 0, -2, 0, 0, 2, 0, 0, 4, 0, 0, 8, 0, 0,
    -8, 2, 0, -6, 2, 0, -2, 2, 0, 0, 2, 0, 4, 2, 0, 6, 2, 0, -10, 4,
    0, -6, 4, 0, -4, 4, 0, 0, 4, 0, 2, 4, 0, 6, 4, 0, -10, 6, 0, -8,
    6, 0, -4, 6, 0, -2, 6, 0, 2, 6, 0, 4, 6, 0, -8, 8, 0, -6, 8, 0,
    -2, 8, 0, 0, 8, 0, -6, 10, 0, -4, 10, 0]).reshape((54, 3)) / 18.0

cc54_1x1 = np.array([
    2, 2, 0, 4, 4, 0, 8, 8, 0, 6, 8, 0, 4, 6, 0, 6,
    10, 0, 4, 10, 0, 2, 6, 0, 2, 8, 0, 0, 2, 0, 0, 4, 0, 0, 8, 0, -2,
    6, 0, -2, 4, 0, -4, 6, 0, -6, 4, 0, -4, 2, 0, -6, 2, 0, -2, 0, 0,
    -4, 0, 0, -8, 0, 0, -8, -2, 0, -6, -2, 0, -10, -4, 0, -10, -6, 0,
    -6, -4, 0, -8, -6, 0, -2, -2, 0, -4, -4, 0, -8, -8, 0, 4, -2, 0,
    6, -2, 0, 6, -4, 0, 2, 0, 0, 4, 0, 0, 6, 2, 0, 6, 4, 0, 8, 6, 0,
    8, 0, 0, 8, 2, 0, 10, 4, 0, 10, 6, 0, 2, -4, 0, 2, -6, 0, 4, -6,
    0, 0, -2, 0, 0, -4, 0, -2, -6, 0, -4, -6, 0, -6, -8, 0, 0, -8, 0,
    -2, -8, 0, -4, -10, 0, -6, -10, 0]).reshape((54, 3)) / 18.0

cc162_sq3xsq3 = np.array([
    -8, 16, 0, -10, 14, 0, -7, 14, 0, -4, 14,
    0, -11, 13, 0, -8, 13, 0, -5, 13, 0, -2, 13, 0, -13, 11, 0, -10,
    11, 0, -7, 11, 0, -4, 11, 0, -1, 11, 0, 2, 11, 0, -14, 10, 0, -11,
    10, 0, -8, 10, 0, -5, 10, 0, -2, 10, 0, 1, 10, 0, 4, 10, 0, -16,
    8, 0, -13, 8, 0, -10, 8, 0, -7, 8, 0, -4, 8, 0, -1, 8, 0, 2, 8, 0,
    5, 8, 0, 8, 8, 0, -14, 7, 0, -11, 7, 0, -8, 7, 0, -5, 7, 0, -2, 7,
    0, 1, 7, 0, 4, 7, 0, 7, 7, 0, 10, 7, 0, -13, 5, 0, -10, 5, 0, -7,
    5, 0, -4, 5, 0, -1, 5, 0, 2, 5, 0, 5, 5, 0, 8, 5, 0, 11, 5, 0,
    -14, 4, 0, -11, 4, 0, -8, 4, 0, -5, 4, 0, -2, 4, 0, 1, 4, 0, 4, 4,
    0, 7, 4, 0, 10, 4, 0, -13, 2, 0, -10, 2, 0, -7, 2, 0, -4, 2, 0,
    -1, 2, 0, 2, 2, 0, 5, 2, 0, 8, 2, 0, 11, 2, 0, -11, 1, 0, -8, 1,
    0, -5, 1, 0, -2, 1, 0, 1, 1, 0, 4, 1, 0, 7, 1, 0, 10, 1, 0, 13, 1,
    0, -10, -1, 0, -7, -1, 0, -4, -1, 0, -1, -1, 0, 2, -1, 0, 5, -1,
    0, 8, -1, 0, 11, -1, 0, 14, -1, 0, -11, -2, 0, -8, -2, 0, -5, -2,
    0, -2, -2, 0, 1, -2, 0, 4, -2, 0, 7, -2, 0, 10, -2, 0, 13, -2, 0,
    -10, -4, 0, -7, -4, 0, -4, -4, 0, -1, -4, 0, 2, -4, 0, 5, -4, 0,
    8, -4, 0, 11, -4, 0, 14, -4, 0, -8, -5, 0, -5, -5, 0, -2, -5, 0,
    1, -5, 0, 4, -5, 0, 7, -5, 0, 10, -5, 0, 13, -5, 0, 16, -5, 0, -7,
    -7, 0, -4, -7, 0, -1, -7, 0, 2, -7, 0, 5, -7, 0, 8, -7, 0, 11, -7,
    0, 14, -7, 0, 17, -7, 0, -8, -8, 0, -5, -8, 0, -2, -8, 0, 1, -8,
    0, 4, -8, 0, 7, -8, 0, 10, -8, 0, 13, -8, 0, 16, -8, 0, -7, -10,
    0, -4, -10, 0, -1, -10, 0, 2, -10, 0, 5, -10, 0, 8, -10, 0, 11,
    -10, 0, 14, -10, 0, 17, -10, 0, -5, -11, 0, -2, -11, 0, 1, -11, 0,
    4, -11, 0, 7, -11, 0, 10, -11, 0, 13, -11, 0, 16, -11, 0, -1, -13,
    0, 2, -13, 0, 5, -13, 0, 8, -13, 0, 11, -13, 0, 14, -13, 0, 1,
    -14, 0, 4, -14, 0, 7, -14, 0, 10, -14, 0, 13, -14, 0, 5, -16, 0,
    8, -16, 0, 11, -16, 0, 7, -17, 0, 10, -17, 0]).reshape((162, 3)) / 27.0

cc162_1x1 = np.array([
    -8, -16, 0, -10, -14, 0, -7, -14, 0, -4, -14,
    0, -11, -13, 0, -8, -13, 0, -5, -13, 0, -2, -13, 0, -13, -11, 0,
    -10, -11, 0, -7, -11, 0, -4, -11, 0, -1, -11, 0, 2, -11, 0, -14,
    -10, 0, -11, -10, 0, -8, -10, 0, -5, -10, 0, -2, -10, 0, 1, -10,
    0, 4, -10, 0, -16, -8, 0, -13, -8, 0, -10, -8, 0, -7, -8, 0, -4,
    -8, 0, -1, -8, 0, 2, -8, 0, 5, -8, 0, 8, -8, 0, -14, -7, 0, -11,
    -7, 0, -8, -7, 0, -5, -7, 0, -2, -7, 0, 1, -7, 0, 4, -7, 0, 7, -7,
    0, 10, -7, 0, -13, -5, 0, -10, -5, 0, -7, -5, 0, -4, -5, 0, -1,
    -5, 0, 2, -5, 0, 5, -5, 0, 8, -5, 0, 11, -5, 0, -14, -4, 0, -11,
    -4, 0, -8, -4, 0, -5, -4, 0, -2, -4, 0, 1, -4, 0, 4, -4, 0, 7, -4,
    0, 10, -4, 0, -13, -2, 0, -10, -2, 0, -7, -2, 0, -4, -2, 0, -1,
    -2, 0, 2, -2, 0, 5, -2, 0, 8, -2, 0, 11, -2, 0, -11, -1, 0, -8,
    -1, 0, -5, -1, 0, -2, -1, 0, 1, -1, 0, 4, -1, 0, 7, -1, 0, 10, -1,
    0, 13, -1, 0, -10, 1, 0, -7, 1, 0, -4, 1, 0, -1, 1, 0, 2, 1, 0, 5,
    1, 0, 8, 1, 0, 11, 1, 0, 14, 1, 0, -11, 2, 0, -8, 2, 0, -5, 2, 0,
    -2, 2, 0, 1, 2, 0, 4, 2, 0, 7, 2, 0, 10, 2, 0, 13, 2, 0, -10, 4,
    0, -7, 4, 0, -4, 4, 0, -1, 4, 0, 2, 4, 0, 5, 4, 0, 8, 4, 0, 11, 4,
    0, 14, 4, 0, -8, 5, 0, -5, 5, 0, -2, 5, 0, 1, 5, 0, 4, 5, 0, 7, 5,
    0, 10, 5, 0, 13, 5, 0, 16, 5, 0, -7, 7, 0, -4, 7, 0, -1, 7, 0, 2,
    7, 0, 5, 7, 0, 8, 7, 0, 11, 7, 0, 14, 7, 0, 17, 7, 0, -8, 8, 0,
    -5, 8, 0, -2, 8, 0, 1, 8, 0, 4, 8, 0, 7, 8, 0, 10, 8, 0, 13, 8, 0,
    16, 8, 0, -7, 10, 0, -4, 10, 0, -1, 10, 0, 2, 10, 0, 5, 10, 0, 8,
    10, 0, 11, 10, 0, 14, 10, 0, 17, 10, 0, -5, 11, 0, -2, 11, 0, 1,
    11, 0, 4, 11, 0, 7, 11, 0, 10, 11, 0, 13, 11, 0, 16, 11, 0, -1,
    13, 0, 2, 13, 0, 5, 13, 0, 8, 13, 0, 11, 13, 0, 14, 13, 0, 1, 14,
    0, 4, 14, 0, 7, 14, 0, 10, 14, 0, 13, 14, 0, 5, 16, 0, 8, 16, 0,
    11, 16, 0, 7, 17, 0, 10, 17, 0]).reshape((162, 3)) / 27.0

# The following is a list of the critical points in the 1. Brillouin zone
# for some typical crystal structures.
# (In units of the reciprocal basis vectors)
# See http://en.wikipedia.org/wiki/Brillouin_zone

ibz_points = {'cubic': {'Gamma': [0, 0, 0],
                        'X': [0, 0 / 2, 1 / 2],
                        'R': [1 / 2, 1 / 2, 1 / 2],
                        'M': [0 / 2, 1 / 2, 1 / 2]},
              'fcc': {'Gamma': [0, 0, 0],
                      'X': [1 / 2, 0, 1 / 2],
                      'W': [1 / 2, 1 / 4, 3 / 4],
                      'K': [3 / 8, 3 / 8, 3 / 4],
                      'U': [5 / 8, 1 / 4, 5 / 8],
                      'L': [1 / 2, 1 / 2, 1 / 2]},
              'bcc': {'Gamma': [0, 0, 0],
                      'H': [1 / 2, -1 / 2, 1 / 2],
                      'N': [0, 0, 1 / 2],
                      'P': [1 / 4, 1 / 4, 1 / 4]},
              'hexagonal': {'Gamma': [0, 0, 0],
                            'M': [0, 1 / 2, 0],
                            'K': [-1 / 3, 1 / 3, 0],
                            'A': [0, 0, 1 / 2],
                            'L': [0, 1 / 2, 1 / 2],
                            'H': [-1 / 3, 1 / 3, 1 / 2]},
              'tetragonal': {'Gamma': [0, 0, 0],
                             'X': [1 / 2, 0, 0],
                             'M': [1 / 2, 1 / 2, 0],
                             'Z': [0, 0, 1 / 2],
                             'R': [1 / 2, 0, 1 / 2],
                             'A': [1 / 2, 1 / 2, 1 / 2]},
              'orthorhombic': {'Gamma': [0, 0, 0],
                               'R': [1 / 2, 1 / 2, 1 / 2],
                               'S': [1 / 2, 1 / 2, 0],
                               'T': [0, 1 / 2, 1 / 2],
                               'U': [1 / 2, 0, 1 / 2],
                               'X': [1 / 2, 0, 0],
                               'Y': [0, 1 / 2, 0],
                               'Z': [0, 0, 1 / 2]}}
