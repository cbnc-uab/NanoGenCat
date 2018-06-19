"""Helper functions for creating supercells."""

import numpy as np


def get_deviation_from_optimal_cell_shape(cell, target_shape='sc', norm=None):
    """Calculate the deviation of the given cell metric from the ideal
    cell metric defining a certain shape. Specifically, the function
    evaluates the expression `\Delta = || Q \mathbf{h} -
    \mathbf{h}_{target}||_2`, where `\mathbf{h}` is the input
    metric (*cell*) and `Q` is a normalization factor (*norm*)
    while the target metric `\mathbf{h}_{target}` (via
    *target_shape*) represent simple cubic ('sc') or face-centered
    cubic ('fcc') cell shapes.

    Parameters:

    cell: 2D array of floats
        Metric given as a (3x3 matrix) of the input structure.
    target_shape: str
        Desired supercell shape. Can be 'sc' for simple cubic or
        'fcc' for face-centered cubic.
    norm: float
        Specify the normalization factor. This is useful to avoid
        recomputing the normalization factor when computing the
        deviation for a series of P matrices.

    """

    if target_shape in ['sc', 'simple-cubic']:
        target_metric = np.eye(3)
    elif target_shape in ['fcc', 'face-centered cubic']:
        target_metric = 0.5 * np.array([[0, 1, 1],
                                        [1, 0, 1],
                                        [1, 1, 0]])
    if not norm:
        norm = (np.linalg.det(cell) /
                np.linalg.det(target_metric))**(-1.0 / 3)
    return np.linalg.norm(norm * cell - target_metric)


def find_optimal_cell_shape(cell, target_size, target_shape,
                            lower_limit=-2, upper_limit=2,
                            verbose=False):
    """Returns the transformation matrix that produces a supercell
    corresponding to a certain number of unit cells (*target_size*)
    based on a primitive metric (*cell*) that most closely
    approximates a desired shape (*target_shape*).

    Note: This implementation uses inline-C via
    `scipy.weave
    <http://docs.scipy.org/doc/scipy/reference/tutorial/weave.html>`_
    to achieve a significant speed-up (about two orders of magnitude)
    compared to the pure python implementation in the
    :func:`~ase.build.find_optimal_cell_shape_pure_python` function.

    Parameters:

    cell: 2D array of floats
        Metric given as a (3x3 matrix) of the input structure.
    target_size: integer
        Size of desired super cell in number of unit cells.
    target_shape: str
        Desired supercell shape. Can be 'sc' for simple cubic or
        'fcc' for face-centered cubic.
    lower_limit: int
        Lower limit of search range.
    upper_limit: int
        Upper limit of search range.
    verbose: bool
        Set to True to obtain additional information regarding
        construction of transformation matrix.

    """

    # Inline C code that does the heavy work.
    # It iterates over all possible matrices and finds the best match.
    code = """
    #include <math.h>

    // The code below will search over a large number of matrices
    // that are generated from a set of integer numbers between
    // imin and imax.
    int imin = search_range[0];
    int imax = search_range[1];

    // For the sake of readability the input cell metric is copied
    // to a set of double variables.
    double h11 = norm_cell[0];   // [3 * 0 + 0];
    double h12 = norm_cell[1];   // [3 * 0 + 1];
    double h13 = norm_cell[2];   // [3 * 0 + 2];
    double h21 = norm_cell[3];   // [3 * 1 + 0];
    double h22 = norm_cell[4];   // [3 * 1 + 1];
    double h23 = norm_cell[5];   // [3 * 1 + 2];
    double h31 = norm_cell[6];   // [3 * 2 + 0];
    double h32 = norm_cell[7];   // [3 * 2 + 1];
    double h33 = norm_cell[8];   // [3 * 2 + 2];

    double det_P;  // will store determinant of P matrix
    double m;      // auxiliary variable
    double current_score; // l2-norm of (QPh - h_opt)

    for (int dxx=imin ; dxx<=imax ; dxx++)
    for (int dxy=imin ; dxy<=imax ; dxy++)
    for (int dxz=imin ; dxz<=imax ; dxz++)
    for (int dyx=imin ; dyx<=imax ; dyx++)
    for (int dyy=imin ; dyy<=imax ; dyy++)
    for (int dyz=imin ; dyz<=imax ; dyz++)
    for (int dzx=imin ; dzx<=imax ; dzx++)
    for (int dzy=imin ; dzy<=imax ; dzy++)
    for (int dzz=imin ; dzz<=imax ; dzz++) {

      // P matrix
      int xx = starting_P[0] + dxx;
      int xy = starting_P[1] + dxy;
      int xz = starting_P[2] + dxz;
      int yx = starting_P[3] + dyx;
      int yy = starting_P[4] + dyy;
      int yz = starting_P[5] + dyz;
      int zx = starting_P[6] + dzx;
      int zy = starting_P[7] + dzy;
      int zz = starting_P[8] + dzz;

      // compute determinant
      det_P = 0;
      det_P += xx*yy*zz;
      det_P += xy*yz*zx;
      det_P += xz*yx*zy;
      det_P -= xx*yz*zy;
      det_P -= xy*yx*zz;
      det_P -= xz*yy*zx;

      if (det_P == target_size[0]) {

        // compute l2-norm squared (taking the square root
        // is unnecessary and just consumes computer time)
        current_score = 0.0;
        current_score += pow(xx * h11 + xy * h21 + xz * h31 - target_metric[0],
                             2); // 1-1
        current_score += pow(xx * h12 + xy * h22 + xz * h32 - target_metric[1],
                             2);
        current_score += pow(xx * h13 + xy * h23 + xz * h33 - target_metric[2],
                             2);
        current_score += pow(yx * h11 + yy * h21 + yz * h31 - target_metric[3],
                             2);
        current_score += pow(yx * h12 + yy * h22 + yz * h32 - target_metric[4],
                             2); // 2-2
        current_score += pow(yx * h13 + yy * h23 + yz * h33 - target_metric[5],
                             2);
        current_score += pow(zx * h11 + zy * h21 + zz * h31 - target_metric[6],
                             2);
        current_score += pow(zx * h12 + zy * h22 + zz * h32 - target_metric[7],
                             2);
        current_score += pow(zx * h13 + zy * h23 + zz * h33 - target_metric[8],
                             2); // 3-3

        if (current_score < best_score[0]) {
          best_score[0] = current_score;
          optimal_P[0] = xx;   // [3 * 0 + 0];
          optimal_P[1] = xy;   // [3 * 0 + 1];
          optimal_P[2] = xz;   // [3 * 0 + 2];
          optimal_P[3] = yx;   // [3 * 1 + 0];
          optimal_P[4] = yy;   // [3 * 1 + 1];
          optimal_P[5] = yz;   // [3 * 1 + 2];
          optimal_P[6] = zx;   // [3 * 2 + 0];
          optimal_P[7] = zy;   // [3 * 2 + 1];
          optimal_P[8] = zz;   // [3 * 2 + 2];
        }
      }
    }
    """

    # Set up target metric
    if target_shape in ['sc', 'simple-cubic']:
        target_metric = np.eye(3)
    elif target_shape in ['fcc', 'face-centered cubic']:
        target_metric = 0.5 * np.array([[0, 1, 1],
                                        [1, 0, 1],
                                        [1, 1, 0]], dtype=float)
    if verbose:
        print('target metric (h_target):')
        print(target_metric)

    # Normalize cell metric to reduce computation time during looping
    norm = (target_size * np.linalg.det(cell) /
            np.linalg.det(target_metric))**(-1.0 / 3)
    norm_cell = norm * cell
    if verbose:
        print('normalization factor (Q): %g' % norm)

    # Approximate initial P matrix
    ideal_P = np.dot(target_metric, np.linalg.inv(norm_cell))
    if verbose:
        print('idealized transformation matrix:')
        print(ideal_P)
    starting_P = np.array(np.around(ideal_P, 0), dtype=int)
    if verbose:
        print('closest integer transformation matrix (P_0):')
        print(starting_P)

    # Prepare run.
    # Weave expects all input/output variables to be numpy arrays.
    best_score = np.array([100000.0])
    optimal_P = np.zeros((3, 3))
    search_range = np.array([lower_limit, upper_limit], dtype=int)
    target_size = np.array(target_size, dtype=int)
    # Execute the inline C code.
    from scipy import weave
    weave.inline(code, ['search_range', 'norm_cell', 'best_score',
                        'optimal_P', 'target_metric', 'target_size',
                        'starting_P'])

    # This is done to satisfy pyflakes/pep8.
    search_range -= 1

    # Finalize.
    if verbose:
        print('smallest score (|Q P h_p - h_target|_2): %f' % best_score)
        print('optimal transformation matrix (P_opt):')
        print(optimal_P)
        print('supercell metric:')
        print(np.round(np.dot(optimal_P, cell), 4))
        print('determinant of optimal transformation matrix: %d' %
              np.linalg.det(optimal_P))
    return optimal_P


def find_optimal_cell_shape_pure_python(cell, target_size, target_shape,
                                        lower_limit=-2, upper_limit=2,
                                        verbose=False):
    """Returns the transformation matrix that produces a supercell
    corresponding to *target_size* unit cells with metric *cell* that
    most closely approximates the shape defined by *target_shape*.

    Note: This pure python implementation of the is much slower than
    the inline-C version provided in
    :func:`~ase.build.find_optimal_cell_shape`.

    Parameters:

    cell: 2D array of floats
        Metric given as a (3x3 matrix) of the input structure.
    target_size: integer
        Size of desired super cell in number of unit cells.
    target_shape: str
        Desired supercell shape. Can be 'sc' for simple cubic or
        'fcc' for face-centered cubic.
    lower_limit: int
        Lower limit of search range.
    upper_limit: int
        Upper limit of search range.
    verbose: bool
        Set to True to obtain additional information regarding
        construction of transformation matrix.

    """

    # Set up target metric
    if target_shape in ['sc', 'simple-cubic']:
        target_metric = np.eye(3)
    elif target_shape in ['fcc', 'face-centered cubic']:
        target_metric = 0.5 * np.array([[0, 1, 1],
                                        [1, 0, 1],
                                        [1, 1, 0]], dtype=float)
    if verbose:
        print('target metric (h_target):')
        print(target_metric)

    # Normalize cell metric to reduce computation time during looping
    norm = (target_size * np.linalg.det(cell) /
            np.linalg.det(target_metric))**(-1.0 / 3)
    norm_cell = norm * cell
    if verbose:
        print('normalization factor (Q): %g' % norm)

    # Approximate initial P matrix
    ideal_P = np.dot(target_metric, np.linalg.inv(norm_cell))
    if verbose:
        print('idealized transformation matrix:')
        print(ideal_P)
    starting_P = np.array(np.around(ideal_P, 0), dtype=int)
    if verbose:
        print('closest integer transformation matrix (P_0):')
        print(starting_P)

    # Prepare run.
    from itertools import product
    best_score = 1e6
    optimal_P = None
    for dP in product(range(lower_limit, upper_limit + 1), repeat=9):
        dP = np.array(dP, dtype=int).reshape(3, 3)
        P = starting_P + dP
        if int(np.linalg.det(P)) != target_size:
            continue
        score = get_deviation_from_optimal_cell_shape(
            np.dot(P, norm_cell), target_shape=target_shape, norm=1.0)
        if score < best_score:
            best_score = score
            optimal_P = P

    if optimal_P is None:
        print('Failed to find a transformation matrix.')
        return None

    # Finalize.
    if verbose:
        print('smallest score (|Q P h_p - h_target|_2): %f' % best_score)
        print('optimal transformation matrix (P_opt):')
        print(optimal_P)
        print('supercell metric:')
        print(np.round(np.dot(optimal_P, cell), 4))
        print('determinant of optimal transformation matrix: %d' %
              np.linalg.det(optimal_P))
    return optimal_P


def make_supercell(prim, P):
    """Generate a supercell by applying a general transformation (*P*) to
    the input configuration (*prim*).

    The transformation is described by a 3x3 integer matrix
    `\mathbf{P}`. Specifically, the new cell metric
    `\mathbf{h}` is given in terms of the metric of the input
    configuraton `\mathbf{h}_p` by `\mathbf{P h}_p =
    \mathbf{h}`.

    Internally this function uses the :func:`~ase.build.cut` function.

    Parameters:

    prim: ASE Atoms object
        Input configuration.
    P: 3x3 integer matrix
        Transformation matrix `\mathbf{P}`.

    """

    from ase.build import cut
    return cut(prim, P[0], P[1], P[2])
