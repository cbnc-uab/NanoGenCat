from ase.spacegroup import crystal
from ase.visualize import view
# from ase.cluster import wulff_construction
from ase.build import cut
from ase.io import  write, read
import numpy as np
import sys
import os

sys.path.append(os.path.abspath("bcnm/"))
from bcn_wulff import bcn_wulff_construction

a=8.3958

magnetite=crystal(['Ti','Fe','O'], basis=[(0.1250,0.1250,0.1250),(0.5,0.5,0.5),(0.25470,0.25470,0.25470)],spacegroup=227,
        cellpar=[a,a,a,90,90,90],primitive_cell=False)



# view(magnetite)
surfaces = [(0,0,1),(1,1,1)]
esurf=[0.96,1.09]
for size in np.arange(20,30,1):
# size=24
	atoms = bcn_wulff_construction(magnetite,surfaces,esurf,float(size),'ext',rounding='above',debug=1,option=1)
