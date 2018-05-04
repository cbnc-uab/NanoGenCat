import numpy as np

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize.precon import Exp, PreconLBFGS, PreconFIRE

N = 1
a0 = bulk('Cu', cubic=True)
a0 *= (N, N, N)

# perturb the atoms
s = a0.get_scaled_positions()
s[:, 0] *= 0.995
a0.set_scaled_positions(s)

nsteps = []
energies = []
for OPT in [PreconLBFGS, PreconFIRE]:
    for precon in [None, Exp(A=3, use_pyamg=False)]:
        atoms = a0.copy()
        atoms.set_calculator(EMT())
        opt = OPT(atoms, precon=precon, use_armijo=True)
        opt.run(1e-4)
        energies += [atoms.get_potential_energy()]
        nsteps += [opt.get_number_of_steps()]

# check we get the expected energy for all methods
assert np.abs(np.array(energies) - -0.022726045433998365).max() < 1e-4
