from ase.spacegroup import crystal
from ase.visualize import view
# from ase.cluster import wulff_construction
from ase.build import cut
from ase.io import  write, read
import numpy as np
sys.path.append(os.path.abspath("bcnm/"))
from bcn_wulff import bcn_wulff_construction

a=8.3958

magnetite=crystal(['Ti','Fe','O'], basis=[(0.1250,0.1250,0.1250),(0.5,0.5,0.5),(0.25470,0.25470,0.25470)],spacegroup=227,
        setting=2,onduplicates='replace',cellpar=[a,a,a,90,90,90],primitive_cell=False)



view(magnetite)
# surfaces = [(1,1,0),(0,1,1),(1,0,0),(0,0,1)]
# esurf=[0.94,1.06,1.23,1.55]
# # # # for size in np.arange(15,20,1):
# size= 14
# atoms = bcn_wulff_construction(test,surfaces,esurf,float(size),'ext',rounding='above',debug=1,option=1)
# # atoms = wulff_construction(iridium,surfaces,esurf,float(size),'ext',rounding='above',debug=1)

# exit()
# #
