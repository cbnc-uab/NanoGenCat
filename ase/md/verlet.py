import numpy as np

from ase.md.md import MolecularDynamics


class VelocityVerlet(MolecularDynamics):
    def __init__(self, atoms, dt, trajectory=None, logfile=None,
                 loginterval=1):
        MolecularDynamics.__init__(self, atoms, dt, trajectory, logfile,
                                   loginterval)

    def step(self, f):
        p = self.atoms.get_momenta()
        p += 0.5 * self.dt * f
        masses = self.atoms.get_masses()[:, np.newaxis]
        r = self.atoms.get_positions()

        # if we have constraints then this will do the first part of the
        # RATTLE algorithm:
        self.atoms.set_positions(r + self.dt * p / masses)
        if self.atoms.constraints:
            p = (self.atoms.get_positions() - r) * masses / self.dt

        # We need to store the momenta on the atoms before calculating
        # the forces, as in a parallel Asap calculation atoms may
        # migrate during force calculations, and the momenta need to
        # migrate along with the atoms.
        self.atoms.set_momenta(p, apply_constraint=False)

        f = self.atoms.get_forces(md=True)

        # Second part of RATTLE will be done here:
        self.atoms.set_momenta(self.atoms.get_momenta() + 0.5 * self.dt * f)
        return f
