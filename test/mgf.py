from ase.spacegroup import crystal
from ase.visualize import view
# from ase.cluster import wulff_construction
from ase.cluster.bcn_wulff import bcn_wulff_construction
from ase.build import cut
from ase.io import  write, read
import numpy as np

basis=[(0.0,0.0,0.0),(0.303, 0.303, 0.0)]
a= 4.6672  
c= 3.0829   
mgf = crystal('MgF', basis, spacegroup=136, cellpar=[a,a,c,90,90,90],primitive_cell=False)
# print(mgf)
write('crystalShape',mgf,format='xyz')
# view(mgf)

surfaces = [(1,1,0),(1,0,0),(1,0,1),(0,0,1),(1,1,1)]
esurf=[0.67,0.76,0.82,1.07,1.16]
# for size in np.arange(20,30,1):
size= 20.
# atoms = bcn_wulff_construction(mgf,surfaces,esurf,float(size),'ext',rounding='above',debug=1)
# # atoms = wulff_construction(mgf,surfaces,esurf,float(size),'ext',rounding='above',debug=1)
# atoms = wulff_construction(mgf,surfaces,esurf,float(size),'ext',rounding='above')

exit()
# #
