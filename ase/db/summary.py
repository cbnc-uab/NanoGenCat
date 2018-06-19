from __future__ import print_function

from ase.data import atomic_masses, chemical_symbols
from ase.db.core import float_to_time_string, now
from ase.utils import hill


class Summary:
    def __init__(self, row, subscript=None):
        self.row = row

        self.cell = [['{0:.3f}'.format(a) for a in axis] for axis in row.cell]

        forces = row.get('constrained_forces')
        if forces is None:
            fmax = None
            self.forces = None
        else:
            fmax = (forces**2).sum(1).max()**0.5
            N = len(forces)
            self.forces = []
            for n, f in enumerate(forces):
                if n < 5 or n >= N - 5:
                    f = tuple('{0:10.3f}'.format(x) for x in f)
                    symbol = chemical_symbols[row.numbers[n]]
                    self.forces.append((n, symbol) + f)
                elif n == 5:
                    self.forces.append((' ...', '',
                                        '       ...',
                                        '       ...',
                                        '       ...'))

        self.stress = row.get('stress')
        if self.stress is not None:
            self.stress = ', '.join('{0:.3f}'.format(s) for s in self.stress)

        if 'masses' in row:
            mass = row.masses.sum()
        else:
            mass = atomic_masses[row.numbers].sum()

        formula = hill(row.numbers)
        if subscript:
            formula = subscript.sub(r'<sub>\1</sub>', formula)

        table = [
            ('id', '', row.id),
            ('age', '', float_to_time_string(now() - row.ctime, True)),
            ('formula', '', formula),
            ('user', '', row.user),
            ('calculator', '', row.get('calculator')),
            ('energy', 'eV', row.get('energy')),
            ('fmax', 'eV/Ang', fmax),
            ('charge', '|e|', row.get('charge')),
            ('mass', 'au', mass),
            ('magnetic moment', 'au', row.get('magmom')),
            ('unique id', '', row.unique_id),
            ('volume', 'Ang^3', row.get('volume'))]

        self.table = [(name, unit, value) for name, unit, value in table
                      if value is not None]

        self.key_value_pairs = sorted(row.key_value_pairs.items()) or None

        self.dipole = row.get('dipole')
        if self.dipole is not None:
            self.dipole = ', '.join('{0:.3f}'.format(d) for d in self.dipole)

        self.plots = []
        self.data = row.get('data')
        if self.data:
            plots = []
            for name, value in self.data.items():
                if isinstance(value, dict) and 'xlabel' in value:
                    plots.append((value.get('number'), name))
            self.plots = [name for number, name in sorted(plots)]

            self.data = ', '.join(self.data.keys())

        self.constraints = row.get('constraints')
        if self.constraints:
            self.constraints = ', '.join(d['name'] for d in self.constraints)

    def write(self):
        row = self.row

        width = max(len(name) for name, unit, value in self.table)
        print('{0:{width}}|unit  |value'.format('name', width=width))
        for name, unit, value in self.table:
            print('{0:{width}}|{1:6}|{2}'.format(name, unit, value,
                                                 width=width))

        print('\nUnit cell in Ang:')
        print('axis|periodic|          x|          y|          z')
        c = 1
        for p, axis in zip(row.pbc, self.cell):
            print('   {0}|     {1}|{2[0]:>11}|{2[1]:>11}|{2[2]:>11}'.format(
                c, [' no', 'yes'][p], axis))
            c += 1

        if self.key_value_pairs:
            print('\nKey-value pairs:')
            width = max(len(key) for key, value in self.key_value_pairs)
            for key, value in self.key_value_pairs:
                print('{0:{width}}|{1}'.format(key, value, width=width))

        if self.forces:
            print('\nForces in ev/Ang:')
            for f in self.forces:
                print('{0:4}|{1:2}|{2}|{3}|{4}'.format(*f))

        if self.stress:
            print('\nStress tensor (xx, yy, zz, zy, zx, yx) in eV/Ang^3:')
            print('   ', self.stress)

        if self.dipole:
            print('\nDipole moment in e*Ang: ({0})'.format(self.dipole))

        if self.constraints:
            print('\nConstraints:', self.constraints)

        if self.data:
            print('\nData:', self.data)
