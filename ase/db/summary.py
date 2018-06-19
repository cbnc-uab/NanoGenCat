from __future__ import print_function
import os
import shutil
import os.path as op

from ase.data import atomic_masses, chemical_symbols
from ase.db.core import float_to_time_string, now
from ase.geometry import cell_to_cellpar
from ase.utils import formula_metal, Lock


class Summary:
    def __init__(self, row, meta={}, subscript=None, prefix='', tmpdir='.'):
        self.row = row

        self.cell = [['{:.3f}'.format(a) for a in axis] for axis in row.cell]
        par = ['{:.3f}'.format(x) for x in cell_to_cellpar(row.cell)]
        self.lengths = par[:3]
        self.angles = par[3:]

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

        self.formula = formula_metal(row.numbers)

        if subscript:
            self.formula = subscript.sub(r'<sub>\1</sub>', self.formula)

        age = float_to_time_string(now() - row.ctime, True)

        table = dict((key, value)
                     for key, value in [
                         ('id', row.id),
                         ('age', age),
                         ('formula', self.formula),
                         ('user', row.user),
                         ('calculator', row.get('calculator')),
                         ('energy', row.get('energy')),
                         ('fmax', fmax),
                         ('charge', row.get('charge')),
                         ('mass', mass),
                         ('magmom', row.get('magmom')),
                         ('unique id', row.unique_id),
                         ('volume', row.get('volume'))]
                     if value is not None)

        table.update(row.key_value_pairs)

        for key, value in table.items():
            if isinstance(value, float):
                table[key] = '{:.3f}'.format(value)

        kd = meta.get('key_descriptions', {})

        misc = set(table.keys())
        self.layout = []
        for headline, columns in meta['layout']:
            empty = True
            newcolumns = []
            for column in columns:
                newcolumn = []
                for block in column:
                    if block is None:
                        pass
                    elif isinstance(block, tuple):
                        title, keys = block
                        rows = []
                        for key in keys:
                            value = table.get(key, None)
                            if value is not None:
                                if key in misc:
                                    misc.remove(key)
                                desc, unit = kd.get(key, [0, key, ''])[1:]
                                rows.append((desc, value, unit))
                        if rows:
                            block = (title, rows)
                        else:
                            continue
                    elif block.endswith('.png'):
                        name = op.join(tmpdir, prefix + block)
                        if not op.isfile(name):
                            self.create_figures(row, prefix, tmpdir,
                                                meta['functions'])
                        if op.getsize(name) == 0:
                            # Skip empty files:
                            block = None

                    newcolumn.append(block)
                    if block is not None:
                        empty = False
                newcolumns.append(newcolumn)

            if not empty:
                self.layout.append((headline, newcolumns))

        if misc:
            rows = []
            for key in sorted(misc):
                value = table[key]
                desc, unit = kd.get(key, [0, key, ''])[1:]
                rows.append((desc, value, unit))
            self.layout.append(('Miscellaneous', [[('Items', rows)]]))

        self.dipole = row.get('dipole')
        if self.dipole is not None:
            self.dipole = ', '.join('{0:.3f}'.format(d) for d in self.dipole)

        self.data = row.get('data')
        if self.data:
            self.data = ', '.join(self.data.keys())

        self.constraints = row.get('constraints')
        if self.constraints:
            self.constraints = ', '.join(d['name'] for d in self.constraints)

    def create_figures(self, row, prefix, tmpdir, functions):
        with Lock('ase.db.web.lock'):
            for func, filenames in functions:
                for filename in filenames:
                    try:
                        os.remove(filename)
                    except OSError:  # Python 3 only: FileNotFoundError
                        pass
                func(row)
                for filename in filenames:
                    path = os.path.join(tmpdir, prefix + filename)
                    if os.path.isfile(filename):
                        shutil.move(filename, path)
                    else:
                        # Create an empty file:
                        with open(path, 'w'):
                            pass

    def write(self):
        row = self.row

        print(self.formula + ':')
        for headline, columns in self.layout:
            blocks = columns[0]
            if len(columns) == 2:
                blocks += columns[1]
            print((' ' + headline + ' ').center(78, '='))
            for block in blocks:
                if block is None:
                    pass
                elif isinstance(block, tuple):
                    title, keys = block
                    print(title + ':')
                    if not keys:
                        print()
                        continue
                    width = max(len(name) for name, value, unit in keys)
                    print('{:{width}}|value'.format('name', width=width))
                    for name, value, unit in keys:
                        print('{:{width}}|{} {}'.format(name, value, unit,
                                                        width=width))
                    print()
                elif block.endswith('.png'):
                    if op.isfile(block) and op.getsize(block) > 0:
                        print(block)
                    print()
                elif block == 'CELL':
                    print('Unit cell in Ang:')
                    print('axis|periodic|          x|          y|          z')
                    c = 1
                    fmt = '   {0}|     {1}|{2[0]:>11}|{2[1]:>11}|{2[2]:>11}'
                    for p, axis in zip(row.pbc, self.cell):
                        print(fmt.format(c, [' no', 'yes'][p], axis))
                        c += 1
                    print('Lengths: {:>10}{:>10}{:>10}'
                          .format(*self.lengths))
                    print('Angles:  {:>10}{:>10}{:>10}\n'
                          .format(*self.angles))
                elif block == 'FORCES' and self.forces is not None:
                    print('\nForces in ev/Ang:')
                    for f in self.forces:
                        print('{:4}|{:2}|{}|{}|{}'.format(*f))
                    print()

        if self.stress:
            print('Stress tensor (xx, yy, zz, zy, zx, yx) in eV/Ang^3:')
            print('   ', self.stress, '\n')

        if self.dipole:
            print('Dipole moment in e*Ang: ({})\n'.format(self.dipole))

        if self.constraints:
            print('Constraints:', self.constraints, '\n')

        if self.data:
            print('Data:', self.data, '\n')
