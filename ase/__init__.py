# Copyright 2008, 2009 CAMd
# (see accompanying license files for details).

"""Atomic Simulation Environment."""

from distutils.version import LooseVersion

import numpy as np

from ase.atom import Atom
from ase.atoms import Atoms

__all__ = ['Atoms', 'Atom']
__version__ = '3.13.0'

# import ase.parallel early to avoid circular import problems when
# ase.parallel does "from gpaw.mpi import world":
import ase.parallel  # noqa
ase.parallel  # silence pyflakes

if LooseVersion(np.__version__) < '1.9':
    # Make isinstance(x, numbers.Integral) work also for np.intxx:
    import numbers
    numbers.Integral.register(np.integer)
    del numbers
