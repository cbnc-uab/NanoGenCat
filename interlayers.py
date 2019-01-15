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
from ase.build import cut, bulk
from ase.io import  write, read

####
sys.path.append(os.path.abspath("bcnm/"))
from bcn_wulff import bcn_wulff_construction

print('''
BcnM  Copyright (C) 2017 Computational BioNanoCat Group at UAB
This program comes with ABSOLUTELY NO WARRANTY; for details type 'python bcnm.py --help'. 
This is free software, and you are welcome to redistribute it under certain conditions; type 'cat LICENSE' for details.
''')

parser = ArgumentParser(description='Bulk nanoparticles cutter')
parser.add_argument('input', help='Bcnm input file in yaml')
args = parser.parse_args()

# Reading data
with open(args.input,'r') as file:
 data = load(file)

file.close()

#
os.chdir('tmp')
crystalObject = crystal(data['chemicalSpecie'], data['basis'], spacegroup=data['spaceGroupNumber'], cellpar=data['cellDimension'],primitive_cell=False)
write('crystalShape.out',crystalObject,format='xyz')

# Centering
if data['centering'] == 'none':
    shifts = [[0.0, 0.0, 0.0]]

elif data['centering'] == 'onlyCenter':
    shifts = [[0.5, 0.5, 0.5]]

elif data['centering'] == 'centerOfMass':
    # Check if works correctly
    shifts = [crystalObject.get_center_of_mass(scaled= True)]

elif data['centering'] == 'bound':
    shift = []
    shifts = []
    for coordinate in data['basis']:
        shifts.append(coordinate)
    
    for element in range(3):
        shift.append((data['basis'][0][element]+data['basis'][1][element])/2)
    
    shifts.append(shift)
    shifts.append(crystalObject.get_center_of_mass(scaled=True))

elif data['centering'] == 'manualShift': 
    if data['numberOfShifts'] != len(data['shifts']):
        print('Error: numberOfShifts and number shift elements do not match. Example:\nnumberOfShifts: 2\nshifts:\n    - [1, 1, 1]\n    - [1, 1, 1]')
        exit(1)
    else:
        shifts = data['shifts']

elif data ['centering'] == 'nShift':
    if 'nShift' not in data:
        print('Error: nShift parameter is not set. Example:\nnShift: [1, 1, 1]')
        exit(1)

    if len(data['nShift']) != 3:
        print('Error: Invalid amount of shifts. Example:\ncentering:nShift\nshifts: [1, 1, 1]')
        exit(1)

    nShift = data['nShift']
    shifts = []
    for i in range(nShift[0]):
        for j in range(nShift[1]):
            for k in range(nShift[2]):
                shifts.append([float((1/nShift[0])*i),float((1/nShift[1])*j),float((1/nShift[2])*k)])

elif data ['centering'] == 'automatic':
    shifts = []
    #Center of mass
    # print(crystalObject.get_center_of_mass(scaled= True))
    shift=[x for x in crystalObject.get_center_of_mass(scaled= True)]
    shift = []
    shifts.append(shift)
    # Atom center
    for coordinate in data['basis']:
        shifts.append(coordinate)
    #Bond Center
    for element in range(3):
        shift.append((data['basis'][0][element]+data['basis'][1][element])/2)
    shifts.append(shift)


else:
    print('Error: Invalid centering value. Valid options are:\n centering:none\ncentering:onlyCenter\ncentering:centerOfMass\ncentering:manualShift\ncentering:nShift')
    exit(1)

# for shift in shifts:
#     print(shift)
#     newPath = str(shift[0])+'_'+str(shift[1])+'_'+str(shift[2])
#     if not os.path.exists(newPath):
#         os.makedirs(newPath)
#     os.chdir(newPath)
#     atoms = bcn_wulff_construction(crystalObject,data['surfaces'],data['surfaceEnergy'],float(data['nanoparticleSize']),'ext',center = shift, rounding='above',debug=1)

print(data['nanoparticleSize'],data['sizeRange'])

min_size = int(data['nanoparticleSize'] - data['sizeRange'])
max_size = int(data['nanoparticleSize'] + data['sizeRange'])
print(min_size,max_size)

## Initial screening of shifts
evaluation=[]
for size in range(min_size, max_size, data['step']):
    for shift in shifts:
        temp=[size,shift]
        temp2=[x for x in bcn_wulff_construction(crystalObject,data['surfaces'],data['surfaceEnergy'],float(size),'ext',center = shift, rounding='above',debug=0,np0=True)]
        # print(temp2)
        temp.extend(temp2)
        evaluation.append(temp)
        # break
    # break
print('evaluation')
# print(evaluation)
#Discard the models that have false inside        
for n,i in enumerate(evaluation):
    # print(i)
    if not 'False' in i:
        print('las buenas')
        print(i)
        size=i[0]
        # print(size)
        shift=i[1]
        # print('hereeee')
        bcn_wulff_construction(crystalObject,data['surfaces'],data['surfaceEnergy'],float(size),'ext',center = shift, rounding='above',debug=0)
# for np0 in evaluation:



# exit(0)

