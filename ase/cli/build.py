import sys

import numpy as np

from ase.db import connect
from ase.build import bulk
from ase.io import read, write
from ase.visualize import view
from ase.build import molecule
from ase.atoms import Atoms, string2symbols
from ase.data import ground_state_magnetic_moments
from ase.data import atomic_numbers, covalent_radii


class CLICommand:
    short_description = 'Build an atom, molecule or bulk structure'

    @staticmethod
    def add_arguments(parser):
        add = parser.add_argument
        add('name', metavar='name/input-file')
        add('output', nargs='?')
        add('-M', '--magnetic-moment',
            metavar='M1,M2,...',
            help='Magnetic moment(s).  '
            'Use "-M 1" or "-M 2.3,-2.3".')
        add('--modify', metavar='...',
            help='Modify atoms with Python statement.  '
            'Example: --modify="atoms.positions[-1,2]+=0.1".')
        add('-V', '--vacuum', type=float,
            help='Amount of vacuum to add around isolated atoms '
            '(in Angstrom).')
        add('-v', '--vacuum0', type=float,
            help='Deprecated.  Use -V or --vacuum instead.')
        add('--unit-cell',
            help='Unit cell.  Examples: "10.0" or "9,10,11" (in Angstrom).')
        add('--bond-length', type=float,
            help='Bond length of dimer in Angstrom.')
        add('-x', '--crystal-structure',
            help='Crystal structure.',
            choices=['sc', 'fcc', 'bcc', 'hcp', 'diamond',
                     'zincblende', 'rocksalt', 'cesiumchloride',
                     'fluorite', 'wurtzite'])
        add('-a', '--lattice-constant', default='',
            help='Lattice constant(s) in Angstrom.')
        add('--orthorhombic', action='store_true',
            help='Use orthorhombic unit cell.')
        add('--cubic', action='store_true',
            help='Use cubic unit cell.')
        add('-r', '--repeat',
            help='Repeat unit cell.  Use "-r 2" or "-r 2,3,1".')
        add('-g', '--gui', action='store_true')
        add('--periodic', action='store_true')

    @staticmethod
    def run(args, parser):
        if args.vacuum0:
            parser.error('Please use -V or --vacuum instead!')

        if '.' in args.name:
            # Read from file:
            atoms = read(args.name)
        elif args.crystal_structure:
            atoms = build_bulk(args)
        else:
            atoms = build_molecule(args)

        if args.magnetic_moment:
            magmoms = np.array(
                [float(m) for m in args.magnetic_moment.split(',')])
            atoms.set_initial_magnetic_moments(
                np.tile(magmoms, len(atoms) // len(magmoms)))

        if args.modify:
            exec(args.modify, {'atoms': atoms})

        if args.repeat is not None:
            r = args.repeat.split(',')
            if len(r) == 1:
                r = 3 * r
            atoms = atoms.repeat([int(c) for c in r])

        if args.gui:
            view(atoms)

        if args.output:
            write(args.output, atoms)
        elif sys.stdout.isatty():
            write(args.name + '.json', atoms)
        else:
            con = connect(sys.stdout, type='json')
            con.write(atoms, name=args.name)


def build_molecule(args):
    try:
        # Known molecule or atom?
        atoms = molecule(args.name)
    except NotImplementedError:
        symbols = string2symbols(args.name)
        if len(symbols) == 1:
            Z = atomic_numbers[symbols[0]]
            magmom = ground_state_magnetic_moments[Z]
            atoms = Atoms(args.name, magmoms=[magmom])
        elif len(symbols) == 2:
            # Dimer
            if args.bond_length is None:
                b = (covalent_radii[atomic_numbers[symbols[0]]] +
                     covalent_radii[atomic_numbers[symbols[1]]])
            else:
                b = args.bond_length
            atoms = Atoms(args.name, positions=[(0, 0, 0),
                                                (b, 0, 0)])
        else:
            raise ValueError('Unknown molecule: ' + args.name)
    else:
        if len(atoms) == 2 and args.bond_length is not None:
            atoms.set_distance(0, 1, args.bond_length)

    if args.unit_cell is None:
        if args.vacuum:
            atoms.center(vacuum=args.vacuum)
        else:
            atoms.center(about=[0, 0, 0])
    else:
        a = [float(x) for x in args.unit_cell.split(',')]
        if len(a) == 1:
            cell = [a[0], a[0], a[0]]
        elif len(a) == 3:
            cell = a
        else:
            a, b, c, alpha, beta, gamma = a
            degree = np.pi / 180.0
            cosa = np.cos(alpha * degree)
            cosb = np.cos(beta * degree)
            sinb = np.sin(beta * degree)
            cosg = np.cos(gamma * degree)
            sing = np.sin(gamma * degree)
            cell = [[a, 0, 0],
                    [b * cosg, b * sing, 0],
                    [c * cosb, c * (cosa - cosb * cosg) / sing,
                     c * np.sqrt(
                        sinb**2 - ((cosa - cosb * cosg) / sing)**2)]]
        atoms.cell = cell
        atoms.center()

    atoms.pbc = args.periodic

    return atoms


def build_bulk(args):
    L = args.lattice_constant.replace(',', ' ').split()
    d = dict([(key, float(x)) for key, x in zip('ac', L)])
    atoms = bulk(args.name, crystalstructure=args.crystal_structure,
                 a=d.get('a'), c=d.get('c'),
                 orthorhombic=args.orthorhombic, cubic=args.cubic)

    M, X = {'Fe': (2.3, 'bcc'),
            'Co': (1.2, 'hcp'),
            'Ni': (0.6, 'fcc')}.get(args.name, (None, None))
    if M is not None and args.crystal_structure == X:
        atoms.set_initial_magnetic_moments([M] * len(atoms))

    return atoms
