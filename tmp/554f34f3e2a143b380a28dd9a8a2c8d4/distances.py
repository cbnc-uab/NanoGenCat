# coding: utf-8
from ase.io import read
atoms=read('O168Ti84[0.0, 0.0, 0.5]_NP0.xyz')
oxygens=[216,251,114,179]
from scipy.spatial.distance import euclidean
for i in oxygens:
    print(euclidean(atoms.get_center_of_mass(),atoms[i].position))
