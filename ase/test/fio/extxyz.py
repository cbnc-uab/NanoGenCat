# additional tests of the extended XYZ file I/O
# (which is also included in oi.py test case)
# maintainted by James Kermode <james.kermode@gmail.com>

import os

import numpy as np

import ase.io
from ase.atoms import Atoms
from ase.build import bulk

# array data of shape (N, 1) squeezed down to shape (N, ) -- bug fixed
# in commit r4541
at = bulk('Si')
ase.io.write('to.xyz', at, format='extxyz')
at.arrays['ns_extra_data'] = np.zeros((len(at), 1))
assert at.arrays['ns_extra_data'].shape == (2, 1)

ase.io.write('to_new.xyz', at, format='extxyz')
at_new = ase.io.read('to_new.xyz')
assert at_new.arrays['ns_extra_data'].shape == (2,)

os.unlink('to.xyz')
os.unlink('to_new.xyz')

# write sequence of images with different numbers of atoms -- bug fixed
# in commit r4542
images = [at, at * (2, 1, 1), at * (3, 1, 1)]
ase.io.write('multi.xyz', images, format='extxyz')
read_images = ase.io.read('multi.xyz@:')
assert read_images == images
os.unlink('multi.xyz')

# read xyz containing trailing blank line
f = open('structure.xyz', 'w')
f.write("""4
Coordinates
Mg        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016

""")
f.close()
a = ase.io.read('structure.xyz')
os.unlink('structure.xyz')

# read xyz with / and @ signs in key value
f = open('slash.xyz', 'w')
f.write("""4
key1=a key2=a/b key3=a@b key4="a@b"
Mg        -4.25650        3.79180       -2.54123
C         -1.15405        2.86652       -1.26699
C         -5.53758        3.70936        0.63504
C         -7.28250        4.71303       -3.82016
""")
f.close()
a = ase.io.read('slash.xyz')
assert a.info['key1'] == r'a'
assert a.info['key2'] == r'a/b'
assert a.info['key3'] == r'a@b'
assert a.info['key4'] == r'a@b'
os.unlink('slash.xyz')

struct = Atoms('H4', pbc=[True, True, True],
                cell=[[4.00759, 0.0, 0.0], [-2.003795, 3.47067475, 0.0], [3.06349683e-16, 5.30613216e-16, 5.00307]], positions=[[-2.003795e-05, 2.31379473, 0.875437189], [2.00381504, 1.15688001, 4.12763281], [2.00381504, 1.15688001, 3.37697219], [-2.003795e-05, 2.31379473, 1.62609781]])
struct.info = {'key_value_pairs': {'dataset': 'deltatest', 'kpoints': np.array([28, 28, 20]), 'identifier': 'deltatest_H_1.00'}, 'unique_id': '4cf83e2f89c795fb7eaf9662e77542c1'}

ase.io.write('tmp.xyz', struct)
os.unlink('tmp.xyz')
