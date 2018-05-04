from ase.build import bulk
from ase.calculators.test import FreeElectrons
from ase.dft.kpoints import special_paths
from ase.dft.band_structure import BandStructure

a = bulk('Cu')
path = special_paths['fcc']
a.calc = FreeElectrons(nvalence=1,
                       kpts={'path': path, 'npoints': 200})
a.get_potential_energy()
bs = a.calc.band_structure()
print(bs.labels)
bs.write('hmm.json')
bs = BandStructure(filename='hmm.json')
print(bs.labels)
assert ''.join(bs.labels) == 'GXWKGLUWLKUX'
import matplotlib
matplotlib.use('Agg', warn=False)
bs.plot(emax=10, filename='bs.png')
