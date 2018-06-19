from ase.build import molecule
from ase import io
from gpaw import GPAW

txt = 'out.txt'
if 1:
    calculator = GPAW(h=0.3, txt=txt)
    atoms = molecule('H2', calculator=calculator)
    atoms.center(vacuum=3)
    atoms.get_potential_energy()

    atoms.set_initial_magnetic_moments([0.5, 0.5])
    calculator.set(charge=1)
    atoms.get_potential_energy()

# read again
t = io.read(txt, index=':')
assert isinstance(t, list)
assert abs(t[1].get_magnetic_moments() - 0.5).sum() < 1e-14
