from __future__ import print_function
import numpy as np

from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import convert_string_to_fd


class SimpleQMMM(Calculator):
    """Simple QMMM calculator."""
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, selection, qmcalc, mmcalc1, mmcalc2, vacuum=None):
        """SimpleQMMM object.

        The energy is calculated as::
            
                    _          _          _
            E = E  (R  ) - E  (R  ) + E  (R   )
                 QM  QM     MM  QM     MM  all
                 
        parameters:
            
        selection: list of int, slice object or list of bool
            Selection out of all the atoms that belong to the QM part.
        qmcalc: Calculator object
            QM-calculator.
        mmcalc1: Calculator object
            MM-calculator used for QM region.
        mmcalc2: Calculator object
            MM-calculator used for everything.
        vacuum: float or None
            Amount of vacuum to add around QM atoms.  Use None if QM
            calculator doesn't need a box.
            
        """
        self.selection = selection
        self.qmcalc = qmcalc
        self.mmcalc1 = mmcalc1
        self.mmcalc2 = mmcalc2
        self.vacuum = vacuum
        
        self.qmatoms = None
        self.center = None
        
        self.name = '{0}-{1}+{1}'.format(qmcalc.name, mmcalc1.name)
        
        Calculator.__init__(self)
        
    def initialize_qm(self, atoms):
        constraints = atoms.constraints
        atoms.constraints = []
        self.qmatoms = atoms[self.selection]
        atoms.constraints = constraints
        self.qmatoms.pbc = False
        if self.vacuum:
            self.qmatoms.center(vacuum=self.vacuum)
            self.center = self.qmatoms.positions.mean(axis=0)
            
    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        if self.qmatoms is None:
            self.initialize_qm(atoms)
            
        self.qmatoms.positions = atoms.positions[self.selection]
        if self.vacuum:
            self.qmatoms.positions += (self.center -
                                       self.qmatoms.positions.mean(axis=0))
            
        energy = self.mmcalc2.get_potential_energy(atoms)
        forces = self.mmcalc2.get_forces(atoms)
        
        energy += self.qmcalc.get_potential_energy(self.qmatoms)
        qmforces = self.qmcalc.get_forces(self.qmatoms)
        if self.vacuum:
            qmforces -= qmforces.mean(axis=0)
        forces[self.selection] += qmforces
        
        energy -= self.mmcalc1.get_potential_energy(self.qmatoms)
        forces[self.selection] -= self.mmcalc1.get_forces(self.qmatoms)

        self.results['energy'] = energy
        self.results['forces'] = forces

        
class EIQMMM(Calculator):
    """Explicit interaction QMMM calculator."""
    implemented_properties = ['energy', 'forces']
    
    def __init__(self, selection, qmcalc, mmcalc, interaction,
                 vacuum=None, embedding=None, output=None):
        """EIQMMM object.

        The energy is calculated as::
            
                    _          _         _    _
            E = E  (R  ) + E  (R  ) + E (R  , R  )
                 QM  QM     MM  MM     I  QM   MM
                 
        parameters:
            
        selection: list of int, slice object or list of bool
            Selection out of all the atoms that belong to the QM part.
        qmcalc: Calculator object
            QM-calculator.
        mmcalc: Calculator object
            MM-calculator.
        interaction: Interaction object
            Interaction between QM and MM regions.
        vacuum: float or None
            Amount of vacuum to add around QM atoms.  Use None if QM
            calculator doesn't need a box.
        embedding: Embedding object or None
            Specialized embedding object.  Use None in order to use the
            default one.
        output: None, '-', str or file-descriptor.
            File for logging information - default is no logging (None).
            
        """
        
        self.selection = selection

        self.qmcalc = qmcalc
        self.mmcalc = mmcalc
        self.interaction = interaction
        self.vacuum = vacuum
        self.embedding = embedding
        
        self.qmatoms = None
        self.mmatoms = None
        self.mask = None
        self.center = None  # center of QM atoms in QM-box
        
        self.name = '{0}+{1}+{2}'.format(qmcalc.name,
                                         interaction.name,
                                         mmcalc.name)
        
        self.output = convert_string_to_fd(output)
        
        Calculator.__init__(self)
        
    def initialize(self, atoms):
        self.mask = np.zeros(len(atoms), bool)
        self.mask[self.selection] = True
        
        constraints = atoms.constraints
        atoms.constraints = []  # avoid slicing of constraints
        self.qmatoms = atoms[self.mask]
        self.mmatoms = atoms[~self.mask]
        atoms.constraints = constraints
        
        self.qmatoms.pbc = False
        
        if self.vacuum:
            self.qmatoms.center(vacuum=self.vacuum)
            self.center = self.qmatoms.positions.mean(axis=0)
            print('Size of QM-cell after centering:',
                  self.qmatoms.cell.diagonal(), file=self.output)
            
        self.qmatoms.calc = self.qmcalc
        self.mmatoms.calc = self.mmcalc
        
        if self.embedding is None:
            self.embedding = Embedding()
            
        self.embedding.initialize(self.qmatoms, self.mmatoms)
        print('Embedding:', self.embedding, file=self.output)
                        
    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        
        if self.qmatoms is None:
            self.initialize(atoms)
            
        self.mmatoms.set_positions(atoms.positions[~self.mask])
        self.qmatoms.set_positions(atoms.positions[self.mask])
        
        if self.vacuum:
            shift = self.center - self.qmatoms.positions.mean(axis=0)
            self.qmatoms.positions += shift
        else:
            shift = (0, 0, 0)
            
        self.embedding.update(shift)

        ienergy, iqmforces, immforces = self.interaction.calculate(
            self.qmatoms, self.mmatoms, shift)
        
        qmenergy = self.qmatoms.get_potential_energy()
        mmenergy = self.mmatoms.get_potential_energy()
        energy = ienergy + qmenergy + mmenergy
        
        print('Energies: {0:12.3f} {1:+12.3f} {2:+12.3f} = {3:12.3f}'
              .format(ienergy, qmenergy, mmenergy, energy), file=self.output)
        
        qmforces = self.qmatoms.get_forces()
        mmforces = self.mmatoms.get_forces()
        
        mmforces += self.embedding.get_mm_forces()
        
        forces = np.empty((len(atoms), 3))
        forces[self.mask] = qmforces + iqmforces
        forces[~self.mask] = mmforces + immforces

        self.results['energy'] = energy
        self.results['forces'] = forces

        
def wrap(D, cell, pbc):
    """Wrap distances to nearest neighbor (minimum image convention)."""
    for i, periodic in enumerate(pbc):
        if periodic:
            d = D[:, i]
            L = cell[i]
            d[:] = (d + L / 2) % L - L / 2  # modify D inplace
    
    
class Embedding:
    def __init__(self, molecule_size=3, **parameters):
        """Point-charge embedding."""
        self.qmatoms = None
        self.mmatoms = None
        self.molecule_size = molecule_size
        self.parameters = parameters
    
    def __repr__(self):
        return 'Embedding(molecule_size={0})'.format(self.molecule_size)
        
    def initialize(self, qmatoms, mmatoms):
        """Hook up embedding object to QM and MM atomsd objects."""
        self.qmatoms = qmatoms
        self.mmatoms = mmatoms
        self.pcpot = qmatoms.calc.embed(mmatoms.get_initial_charges(),
                                        **self.parameters)
        
    def update(self, shift):
        """Update point-charge positions."""
        # Wrap point-charge positions to the MM-cell closest to the
        # center of the the QM box, but avoid ripping molecules apart:
        qmcenter = self.qmatoms.cell.diagonal() / 2
        n = self.molecule_size
        positions = self.mmatoms.positions.reshape((-1, n, 3)) + shift
        
        # Distances from the center of the QM box to the first atom of
        # each molecule:
        distances = positions[:, 0] - qmcenter
        
        wrap(distances, self.mmatoms.cell.diagonal(), self.mmatoms.pbc)
        offsets = distances - positions[:, 0]
        positions += offsets[:, np.newaxis] + qmcenter
        self.pcpot.set_positions(positions.reshape((-1, 3)))
        
    def get_mm_forces(self):
        """Calculate the forces on the MM-atoms from the QM-part."""
        return self.pcpot.get_forces(self.qmatoms.calc)
        
        
class LJInteractions:
    name = 'LJ'
    
    def __init__(self, parameters):
        """Lennard-Jones type explicit interaction.
        
        parameters: dict
            Mapping from pair of atoms to tuple containing epsilon and sigma
            for that pair.
            
        Example::
            
            lj = LJInteractions({('O', 'O'): (eps, sigma)})
            
        """
        self.parameters = {}
        for (symbol1, symbol2), (epsilon, sigma) in parameters.items():
            Z1 = atomic_numbers[symbol1]
            Z2 = atomic_numbers[symbol2]
            self.parameters[(Z1, Z2)] = epsilon, sigma
            self.parameters[(Z2, Z1)] = epsilon, sigma
        
    def calculate(self, qmatoms, mmatoms, shift):
        qmforces = np.zeros_like(qmatoms.positions)
        mmforces = np.zeros_like(mmatoms.positions)
        species = set(mmatoms.numbers)
        energy = 0.0
        for R1, Z1, F1 in zip(qmatoms.positions, qmatoms.numbers, qmforces):
            for Z2 in species:
                if (Z1, Z2) not in self.parameters:
                    continue
                epsilon, sigma = self.parameters[(Z1, Z2)]
                mask = (mmatoms.numbers == Z2)
                D = mmatoms.positions[mask] + shift - R1
                wrap(D, mmatoms.cell.diagonal(), mmatoms.pbc)
                d2 = (D**2).sum(1)
                c6 = (sigma**2 / d2)**3
                c12 = c6**2
                energy += 4 * epsilon * (c12 - c6).sum()
                f = 24 * epsilon * ((2 * c12 - c6) / d2)[:, np.newaxis] * D
                F1 -= f.sum(0)
                mmforces[mask] += f
        return energy, qmforces, mmforces
