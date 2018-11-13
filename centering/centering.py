from ase.cluster.bcn_wulff import evaluateNp0
import numpy as np
from ase.io import read
atoms=read('Mg59O39[0.0, 0.8, 1.0]_NP0.xyz',format='xyz')
evaluateNp0(atoms)