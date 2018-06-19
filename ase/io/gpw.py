from ase import Atoms
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.singlepoint import SinglePointKPoint
from ase.units import Bohr, Hartree
import ase.io.ulm as ulm
from ase.io.trajectory import read_atoms


def read_gpw(filename):
    try:
        reader = ulm.open(filename)
    except ulm.InvalidULMFileError:
        return read_old_gpw(filename)
    return read_atoms(reader.atoms)


def read_old_gpw(filename):
    from gpaw.io.tar import Reader
    r = Reader(filename)
    positions = r.get('CartesianPositions') * Bohr
    numbers = r.get('AtomicNumbers')
    cell = r.get('UnitCell') * Bohr
    pbc = r.get('BoundaryConditions')
    tags = r.get('Tags')
    magmoms = r.get('MagneticMoments')
    energy = r.get('PotentialEnergy') * Hartree

    if r.has_array('CartesianForces'):
        forces = r.get('CartesianForces') * Hartree / Bohr
    else:
        forces = None

    atoms = Atoms(positions=positions,
                  numbers=numbers,
                  cell=cell,
                  pbc=pbc)
    if tags.any():
        atoms.set_tags(tags)

    if magmoms.any():
        atoms.set_initial_magnetic_moments(magmoms)
        magmom = magmoms.sum()
    else:
        magmoms = None
        magmom = None

    atoms.calc = SinglePointDFTCalculator(atoms, energy=energy,
                                          forces=forces,
                                          magmoms=magmoms,
                                          magmom=magmom)
    kpts = []
    if r.has_array('IBZKPoints'):
        for w, kpt, eps_n, f_n in zip(r.get('IBZKPointWeights'),
                                      r.get('IBZKPoints'),
                                      r.get('Eigenvalues'),
                                      r.get('OccupationNumbers')):
            kpts.append(SinglePointKPoint(w, kpt[0], kpt[1],
                                          eps_n[0], f_n[0]))
    atoms.calc.kpts = kpts

    return atoms
