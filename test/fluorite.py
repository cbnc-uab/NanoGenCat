from ase.spacegroup import crystal
from ase.visualize import view
# from ase.cluster import wulff_construction
from ase.cluster.bcn_wulff import bcn_wulff_construction
from ase.build import cut
from ase.io import  write, read
import numpy as np

basis=[(0.,0.,0.),(0.25, 0.25, 0.25)]
a= 5.411  
caf=crystal('CaF', basis, spacegroup=225, cellpar=[a,a,a,90,90,90],primitive_cell=False)
# print(mgf)
# write('crystalShape',mgf,format='xyz)
# view(caf)

surfaces = [(1,1,1),(1,1,0),(1,0,0)]
esurf=[0.437,0.719,0.979]
# surfaces = [(1,0,0)]
# esurf=[1.54]
# # for size in np.arange(10,30,1):
size= 14.
atoms = bcn_wulff_construction(caf,surfaces,esurf,float(size),'ext',rounding='above',debug=1,option=0)
# # # atoms = wulff_construction(mgf,surfaces,esurf,float(size),'ext',rounding='above',debug=1)
# # atoms = wulff_construction(mgf,surfaces,esurf,float(size),'ext',rounding='above')
#
# exit()
# #
