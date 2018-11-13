from ase.spacegroup import crystal
from ase.visualize import view
# from ase.cluster import wulff_construction
from ase.cluster.bcn_wulff import bcn_wulff_construction
from ase.build import cut
from ase.io import  write, read
import numpy as np

basis=[(0.0,0.0,0.0),(0.25, 0.25, 0.25)]
a= 3.688  
c= 17.149   
mgf = crystal('MgCl', basis, spacegroup=166, cellpar=[a,a,c,90,90,120],primitive_cell=False)

surfaces = [(0,0,1),(0,1,2),(1,0,7),(1,0,4),(0,1,5),(1,1,0)]
esurf=[0.050,0.222,0.183,0.215,0.188,0.264]
for size in np.arange(16,20,1):
#size= 16.
    atoms = bcn_wulff_construction(mgf,surfaces,esurf,float(size),'ext',rounding='above',debug=1,option=1)

exit()
