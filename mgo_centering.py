from ase.spacegroup import crystal
from ase.visualize import view
from ase.cluster.bcn_wulff import bcn_wulff_construction
from ase.build import cut
from ase.calculators import lj
import numpy as np

basis=[(0,0,0), (0.5, 0.5, 0.5)]
a = 3.8
MgO = crystal('MgO', basis, spacegroup=225, cellpar=[a,a,a,90,90,90],primitive_cell=False)


size=10.
surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
esurf=[1.0, 1.14, 0.89]
center = [0.25, 0.25,0.25]
atoms = bcn_wulff_construction(MgO,surfaces,esurf,float(size),'ext',center=center,debug=1)
exit()

