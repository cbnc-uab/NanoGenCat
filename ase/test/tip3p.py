"""Test TIP3P forces."""
from math import cos, sin

from ase import Atoms
from ase.calculators.tip3p import TIP3P, rOH, thetaHOH, set_tip3p_charges

r = rOH
a = thetaHOH

dimer = Atoms('H2OH2O',
              [(r * cos(a), 0, r * sin(a)),
               (r, 0, 0),
               (0, 0, 0),
               (r * cos(a / 2), r * sin(a / 2), 0),
               (r * cos(a / 2), -r * sin(a / 2), 0),
               (0, 0, 0)])
set_tip3p_charges(dimer)
dimer.calc = TIP3P(rc=4.0, width=2.0)  # put O-O distance in the cutoff range
dimer.positions[3:, 0] += 2.8
F = dimer.get_forces()
print(F)
dF = dimer.calc.calculate_numerical_forces(dimer) - F
print(dF)
assert abs(dF).max() < 2e-6
