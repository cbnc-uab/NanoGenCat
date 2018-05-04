from __future__ import print_function
import sys
import functools

import numpy as np


def get_band_gap(calc, direct=False, spin=None, output=sys.stdout):
    """Calculates the band-gap.

    Parameters:

    calc: Calculator object
        Electronic structure calculator object.
    direct: bool
        Calculate direct band-gap.
    spin: int or None
        For spin-polarized systems, you can use spin=0 or spin=1 to look only
        at a single spin-channel.
    output: file descriptor
        Use output=None for no text output.

    Rerurns a (gap, k1, k2) tuple where k1 and k2 are the indices of the
    valence and conduction k-points.  For the spin-polarized case, a
    (gap, (s1, k1), (s2, k2)) tuple is returned.

    Example:

    >>> gap, k1, k2 = get_band_gap(silicon.calc)
    Gap: 1.2 eV
    Transition (v -> c):
        [0.000, 0.000, 0.000] -> [0.500, 0.500, 0.000]
    >>> print(gap, k1, k2)
    1.2 0 5
    >>> gap, k1, k2 = get_band_gap(atoms.calc, direct=True)
    Direct gap: 3.4 eV
    Transition at: [0.000, 0.000, 0.000]
    >>> print(gap, k1, k2)
    3.4 0 0
    """

    kpts_kc = calc.get_ibz_k_points()
    nk = len(kpts_kc)
    ns = calc.get_number_of_spins()
    e_skn = np.array([[calc.get_eigenvalues(kpt=k, spin=s)
                       for k in range(nk)]
                      for s in range(ns)])
    e_skn -= calc.get_fermi_level()
    N_sk = (e_skn < 0.0).sum(2)  # number of occupied bands
    e_skn = np.array([[e_skn[s, k, N_sk[s, k] - 1:N_sk[s, k] + 1]
                       for k in range(nk)]
                      for s in range(ns)])
    ev_sk = e_skn[:, :, 0]  # valence band
    ec_sk = e_skn[:, :, 1]  # conduction band

    if ns == 1:
        gap, k1, k2 = find_gap(N_sk, ev_sk[0], ec_sk[0], direct)
    elif spin is None:
        gap, k1, k2 = find_gap(N_sk, ev_sk.ravel(), ec_sk.ravel(), direct)
        if gap > 0.0:
            k1 = divmod(k1, nk)
            k2 = divmod(k2, nk)
        else:
            k1 = (None, None)
            k2 = (None, None)
    else:
        gap, k1, k2 = find_gap(N_sk[spin:spin + 1], ev_sk[spin], ec_sk[spin],
                               direct)

    if output is not None:
        def sk(k):
            """Convert k or (s, k) to string."""
            if isinstance(k, tuple):
                s, k = k
                return '(spin={0}, {1})'.format(s, sk(k))
            return '[{0:.3f}, {1:.3f}, {2:.3f}]'.format(*kpts_kc[k])

        p = functools.partial(print, file=output)
        if spin is not None:
            p('spin={0}: '.format(spin), end='')
        if gap == 0.0:
            p('No gap!')
        elif direct:
            p('Direct gap: {0:.3f} eV'.format(gap))
            p('Transition at:', sk(k1))
        else:
            p('Gap: {0:.3f} eV'.format(gap))
            p('Transition (v -> c):')
            p('   ', sk(k1), '->', sk(k2))

    return gap, k1, k2


def find_gap(N_sk, ev_k, ec_k, direct):
    """Helper function."""
    if (N_sk.ptp(axis=1) > 0).any():
        # Some band must be crossing the fermi-level
        return 0.0, None, None
    if direct:
        gap_k = ec_k - ev_k
        k = gap_k.argmin()
        return gap_k[k], k, k
    kv = ev_k.argmax()
    kc = ec_k.argmin()
    return ec_k[kc] - ev_k[kv], kv, kc
