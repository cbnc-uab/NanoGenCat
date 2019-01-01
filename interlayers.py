#! /usr/bin/env python3.5
# -*- coding: utf-8 -*-

''' Launcher file for bcnm

'''
import os
import sys
from yaml import load
import numpy as np
from argparse import ArgumentParser
from ase.spacegroup import crystal
from ase.visualize import view
from ase.build import cut
from ase.io import  write, read

####
sys.path.append(os.path.abspath("bcnm/"))
from bcn_wulff import bcn_wulff_construction
from bcn_wulff import interplanarDistance

print('''
BcnM  Copyright (C) 2017 Computational BioNanoCat Group at UAB
This program comes with ABSOLUTELY NO WARRANTY; for details type 'python bcnm.py --help'. 
This is free software, and you are welcome to redistribute it under certain conditions; type 'cat LICENSE' for details.
''')

parser = ArgumentParser(description='Bulk nanoparticles cutter')
parser.add_argument('input', help='Bcnm input file in yaml')
args = parser.parse_args()

with open(args.input,'r') as file:
 data = load(file)

file.close()

os.chdir('tmp')
crystalObject = crystal(data['chemicalSpecie'], data['basis'], spacegroup=data['spaceGroupNumber'], cellpar=data['cellDimension'],primitive_cell=False)
# write('crystalShape.out',crystalObject,format='xyz')
interplanarDistance(crystalObject,data['surfaces'])

# # lista de tuplas
atoms = bcn_wulff_construction(crystalObject,data['surfaces'],data['surfaceEnergy'],float(data['nanoparticleSize']),'ext',rounding='above',debug=1)

# exit()
