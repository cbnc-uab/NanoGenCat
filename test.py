from ase.spacegroup import crystal

from ase.visualize import view
from ase.cluster.cut_cluster import CutCluster as structure

# from ase.build import cut

from ase.io import  write, read

import numpy as np

basis=[(0.0,0.0,0.0),(0.306, 0.306, 0.0)]
a= 4.5424  
c= 3.1862   
iridium = crystal('IrO', basis, spacegroup=136, cellpar=[a,a,c,90,90,90],primitive_cell=False)

# view(iridium)i
surfaces = [(1,1,0),(0,1,1),(1,0,0),(0,0,1)]
distances=None
atoms=structure(iridium,surfaces,10*np.ones(len(surfaces),int),distances, latticeconstant=None)

view(atoms)
# esurf=[0.94,1.06,1.23,1.55]
# # for size in np.arange(20,30,1):
# size= 15.
# atoms = wulff_construction(iridium,surfaces,esurf,float(size),'ext',rounding='above',debug=1)
#
# exit()
#




