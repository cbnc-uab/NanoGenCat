#! /usr/bin/env python3.5
# -*- coding: utf-8 -*-

''' Launcher file for bcnm

'''
import os
import sys
import uuid
import time

from yaml import load, YAMLError
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

from ase.spacegroup import crystal
from ase.visualize import view
from ase.build import cut, bulk
from ase.io import  write, read

from bcnm.bcn_wulff import bcn_wulff_construction, specialCenterings


def main():
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
        try:
            data = load(file)
        except YAMLError as exc:
            if hasattr(exc, 'problem_mark'):
                mark = exc.problem_mark
                print ('Invalid yaml format. Check line: %s of %s ' % (mark.line+1, args.input))
                exit(1)
    file.close()

    # Values by default
    if not 'debug' in data:
        data['debug'] = 0
    if not 'onlyNp0' in data:
        data['onlyNp0'] = False
    if not 'sizeRange' in data:
        data['sizeRange'] = 1
    if not 'wulff-like-method' in data:
        data['wulff-like-method'] = 'surfaceBased'
    if not 'sampleSize' in data:
    	data['sampleSize']=1000
    ####Creating a execution directory
    execDir = Path('tmp/'+str(uuid.uuid4().hex))
    execDir.mkdir(parents=True, exist_ok=True)
    os.chdir(execDir)

    print('Running directory: ',execDir)

    ####printing running parameters
    print('Running parameters')
    for key,values in data.items():
        print(key,':',values)

    ###Start execution
    print('\nStart execution')


    #####Centering
    if data['centering'] == 'none':
        shifts = [[0.0, 0.0, 0.0]]

    elif data['centering'] == 'onlyCenter':
        shifts = [[0.5, 0.5, 0.5]]

    elif data['centering'] == 'centerOfMass':
        ## Making the crystalObject
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
                for C in range(nShift[2]):
                    shifts.append([float((1/nShift[0])*i),float((1/nShift[1])*j),float((1/nShift[2])*k)])

    elif data ['centering'] == 'automatic':
        shifts = []
        #Coordinate origin
        shifts.append([0.,0.,0.])
        # Atom center
        for coordinate in data['basis']:
            shifts.append(coordinate)
        #Xavi proposed positions
        shiftsCenters=specialCenterings(data['spaceGroupNumber'])
        # print (shiftsCenters)
        for shift in shiftsCenters:
            shifts.append(list(shift))
        shifts.append(list(crystalObject.get_center_of_mass(scaled=True)))

    else:
        print('Error: Invalid centering value. Valid options are:\n centering:none\ncentering:onlyCenter\ncentering:centerOfMass\ncentering:manualShift\ncentering:nShift')
        exit(1)


    # print(data['nanoparticleSize'],data['sizeRange'])

    min_size = data['nanoparticleSize'] - data['sizeRange']
    max_size = data['nanoparticleSize'] + data['sizeRange']
    # size=data['nanoparticleSize']
    # print(min_size,max_size)

    ## Initial screening of shifts
    print('\nEvaluation of running parameters on NP0\n')
    startingScreeningTime = time.time()

    print('Size,Shift,Chemical Formula\n')
    evaluation=[]
    for size in np.arange(min_size, max_size, data['step']):
        # if size >8:

	    for shift in shifts:
	        temp=[size,shift]
	        # bcn_wulff_construction(crystalObject,data['surfaces'],data['surfaceEnergy'],float(size),'ext',center = shift, rounding='above',debug=0,np0=True)
	        temp2=[x for x in bcn_wulff_construction(crystalObject,data['surfaces'],data['surfaceEnergy'],float(size),'ext',center = shift, rounding='above',debug=data['debug'],np0=True,wl_method=data['wulff-like-method'])]
	        print(size,shift,temp2[0])
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
    print('Evaluated parameters: Size,Shift,Chemical Formula,Cations, Anions, Minimum coordination, Global coordination,Equivalent planes areas,same order, Wulff-like index')
    print('Results:')
    print(*evaluation, sep='\n')
    print('testing Zone')

    #  getting the booleans as strings, this is done because
    # 0 values are considered as False in the boolean way
    aprovedNp0Models=[]
    for i in evaluation:
        booleans=[str(j) for j in i if str(j)=='True' or str(j)=='False']
        if not 'False' in booleans:
            aprovedNp0Models.append(i)

    # aprovedNp0Models=[i for i in evaluation if not False in i]
    print('\nAproved NP0s:', len(aprovedNp0Models))
    print(*aprovedNp0Models, sep='\n')

    #For each formula keep the one that has the highest total coordination
    # print(aprovedNp0Models[0][2])
    # chemFormList=[i[2] for i in aprovedNp0Models]
    chemicalFormulas=list(set(i[2] for i in aprovedNp0Models))

    #Iterate to get only the one that have the maximum total coordination
    finalModels=[]
    for i in chemicalFormulas:
        np0PerFormula=[]
        for j in aprovedNp0Models:
            if i==j[2]:
                np0PerFormula.append(j)
        tempNp0PerFormula=sorted(np0PerFormula,key=lambda x:x[6],reverse=True)
        # print(tempNp0PerFormula)
        finalModels.append(tempNp0PerFormula[0])

    print('\nFinal NP0s models:',len(finalModels))
    finalSorted=sorted(finalModels,key=lambda x:x[0])
    print(*finalSorted, sep='\n')
    finalScreeningTime = time.time()

    print("Total time evaluation", round(finalScreeningTime-startingScreeningTime)," s")

    if data['onlyNp0']==True:
        exit(0)
    else:
        ##Construction of stoichiometric nanoparticles
        print('\nGenerating stoichiometric nanoparticles\n')
        for i in finalSorted:
            # print(i)
            print('\n NP0: ',i,"\n")
            bcn_wulff_construction(crystalObject,data['surfaces'],data['surfaceEnergy'],
            	float(i[0]),'ext',center = i[1], rounding='above',debug=data['debug'],
            	sampleSize=data['sampleSize'])
        finalTime=time.time()
        print("Total execution time:",round(finalTime-startTime),"s")
        exit(0)

main()
