import sys
import os

####
sys.path.append(os.path.abspath("bcnm/"))

from ase.spacegroup import crystal
from ase.visualize import view
from bcn_wulff import bcn_wulff_construction
from ase.build import cut
from ase.io import  write, read

import numpy as np

# basis=[(0.0,0.0,0.0),(0.306, 0.306, 0.0)]
# a= 4.5424  
# c= 3.1862   
# iridium = crystal('IrO', basis, spacegroup=136, cellpar=[a,a,c,90,90,90],primitive_cell=False)
# print(iridium)
# write('crystalShape',iridium,format='xyz')
# view(iridium)
iridium=read('crystalShape')

surfaces = [(1,1,0),(0,1,1),(1,0,0),(0,0,1)]
esurf=[0.94,1.06,1.23,1.55]
# for size in np.arange(20,30,1):
size= 14
atoms = bcn_wulff_construction(iridium,surfaces,esurf,float(size),'ext',rounding='above',debug=1)

exit()
