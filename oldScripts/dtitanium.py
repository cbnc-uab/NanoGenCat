from ase.spacegroup import crystal
from ase.visualize import view
# from ase.cluster import wulff_construction
from ase.cluster.bcn_wulff import bcn_wulff_construction
from ase.build import cut
from ase.io import  write, read
import numpy as np

basis=[(0.0,0.0,0.0),(0.306, 0.306, 0.0)]
a= 4.603  
c= 2.977   
iridium = crystal('TiO', basis, spacegroup=136, cellpar=[a,a,c,90,90,90],primitive_cell=False)
print(iridium)
write('crystalShape',iridium,format='xyz')
# view(iridium)

surfaces = [(1,1,0),(0,1,1),(1,0,0),(0,0,1)]
esurf=[0.014,0.044,0.022,0.057]
# for size in np.arange(20,30,1):
size= 18
atoms = bcn_wulff_construction(iridium,surfaces,esurf,float(size),'ext',rounding='above',debug=1,option=1)
# atoms = wulff_construction(iridium,surfaces,esurf,float(size),'ext',rounding='above',debug=1)

exit()
#
