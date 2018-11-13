from ase.spacegroup import crystal
from ase.visualize import view
from ase.cluster.bcn_wulff import bcn_wulff_construction,evaluateNp0,reduceNano
from ase.build import cut
from ase.io import read
import numpy as np
from itertools import product
import os


basis=[(0,0,0), (0.5, 0.5, 0.5)]
a = 3.8
MgO = crystal('MgO', basis, spacegroup=225, cellpar=[a,a,a,90,90,90],primitive_cell=False)

centerValues=np.linspace(0,1,5)
print (centerValues)
size=10.
surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
esurf=[1.0, 1.14, 0.89]
# center = [0.25, 0.25,0.25]
# n=0
os.mkdir('centering1')
os.chdir('centering1')
for p in product(centerValues,repeat=3):
	center=list(p)
	filename=str(center[0])+'_'+str(center[1])+'_'+str(center[2])
	os.mkdir(filename)
	os.chdir(filename)
	atoms = bcn_wulff_construction(MgO,surfaces,esurf,float(size),'ext',center=center,option=1)
	os.chdir('../')


exit()

