#! /usr/bin/env python3.5
# -*- coding: utf-8 -*-

''' Launcher file for bcnm

'''
import os
import sys
import uuid
import time

from yaml import load

import numpy as np

from argparse import ArgumentParser


from ase.spacegroup import crystal
from ase.visualize import view
from ase.build import cut, bulk
from ase.io import  write, read

####
sys.path.append(os.path.abspath("bcnm/"))
from bcn_wulff_anatase import bcn_wulff_construction
####
startTime = time.time()

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

####Creating a execution directory
execDir='tmp/'+str(uuid.uuid4().hex)
os.mkdir(execDir)
os.chdir(execDir)
print('Running directory: ',execDir)

####printing running parameters
print('Running parameters')
for key,values in data.items():
    print(key,':',values)

###Start execution
print('\nStart execution')
## Making the crystalObject
crystalObject = crystal(data['chemicalSpecies'], data['basis'], spacegroup=data['spaceGroupNumber'], cellpar=data['cellDimension'],primitive_cell=False)
write('crystalShape.xyz',crystalObject,format='xyz')

##feeding the initial charges 
for atom in crystalObject:
    for n,element in enumerate(data['chemicalSpecies']):
        if atom.symbol==element:
            atom.charge=data['charges'][n]

#####Centering
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
    # Atom center
    for coordinate in data['basis']:
        shifts.append(coordinate)
    #Xavi proposed positions
    shiftsCenters=specialCenterings(data['spaceGroupNumber'])
    # print (shiftsCenters)
    for shift in shiftsCenters:
        shifts.append(list(shift))


else:
    print('Error: Invalid centering value. Valid options are:\n centering:none\ncentering:onlyCenter\ncentering:centerOfMass\ncentering:manualShift\ncentering:nShift')
    exit(1)


# print(data['nanoparticleSize'],data['sizeRange'])

min_size = data['nanoparticleSize'] - data['sizeRange']
max_size = data['nanoparticleSize'] + data['sizeRange']
# print(min_size,max_size)

## Initial screening of shifts
print('\nEvaluation of running parameters on NP0')
startingScreeningTime = time.time()


evaluation=[]
for size in np.arange(min_size, max_size, data['step']):
    # if size >8:
    for shift in shifts:
        print('Size:',size,'Shift:',shift)
        temp=[size,shift]
        # bcn_wulff_construction(crystalObject,data['surfaces'],data['surfaceEnergy'],float(size),'ext',center = shift, rounding='above',debug=0,np0=True)
        temp2=[x for x in bcn_wulff_construction(crystalObject,data['surfaces'],data['surfaceEnergy'],float(size),'ext',center = shift, rounding='above',debug=0,np0=True)]
        # print(temp2)
        temp.extend(temp2)
        evaluation.append(temp)
        # print(temp)
        # break
        # print('Done')
    # else:
    #     print('Size',size,'are too small')
    # break
#Discard the models that have false inside
# print(evaluation)
print('\nNumber of evaluated NP0s: ',len(evaluation))
print('Evaluated parameters: Size,Shift,Chemical Formula,Cations, Anions, Minimum coordination, Global coordination')
print('Results:')
print(*evaluation, sep='\n')

aprovedNp0Models=[i for i in evaluation if not False in i]
print('\nAproved NP0s:', len(aprovedNp0Models))
print(*aprovedNp0Models, sep='\n')

#For each number of metal atoms keep the one with the highest total coordination
#list of unique metal sizes
metalSize=list(set([i[3] for i in aprovedNp0Models]))

#Iterate to get only the one that have the maximum total coordination
finalModels=[]
for i in metalSize:
    np0PerMetal=[]
    for j in aprovedNp0Models:
        if i==j[3]:
            np0PerMetal.append(j)
    tempNp0PerMetal=sorted(np0PerMetal,key=lambda x:x[6],reverse=True)
    # print(tempNp0PerMetal)
    finalModels.append(tempNp0PerMetal[0])

print('\nFinal NP0s models:',len(finalModels))
finalSorted=sorted(finalModels,key=lambda x:x[0],reverse=True)
print(*finalSorted, sep='\n')
finalScreeningTime = time.time()

print("Total time evaluation", round(finalScreeningTime-startingScreeningTime)," s")

if data['onlyNp0']==True:
    exit(0)
else:
    ##Calculation of stoichiometric nanoparticles
    for i in finalModels:
        print('\nGenerating stoichiometric nanoparticles for ',i,"\n")
        bcn_wulff_construction(crystalObject,data['surfaces'],data['surfaceEnergy'],float(i[0]),'ext',center = i[1], rounding='above',debug=0)
    finalTime=time.time()
    print("Total execution time:",round(finalTime-startTime),"s")
    exit(0)
