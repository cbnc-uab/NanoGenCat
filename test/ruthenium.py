from ase.spacegroup import crystal
from ase.visualize import view
from ase.cluster.bcn_wulff import bcn_wulff_construction
# from ase.cluster import wulff_construction
from ase.build import cut
from ase.io import  write, read
import numpy as np

basis=[(0.0,0.0,0.0),(0.306, 0.306, 0.0)]
a= 4.4918  
c= 3.1066   
rutile = crystal('RuO', basis, spacegroup=136, cellpar=[a,a,c,90,90,90],primitive_cell=False)
# print(rutile)
write('crystalShape',rutile,format='xyz')
# view(rutile)
surfaces = [(1,1,0),(0,1,1),(1,0,0),(0,0,1)]
esurf=[1.0574,1.2016,1.314,1.586]
# for size in np.arange(20,30,1):
size= 14.
atoms = bcn_wulff_construction(rutile,surfaces,esurf,float(size),'ext',rounding='above',debug=1)

exit()
