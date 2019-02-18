from __future__ import print_function
import os, time
import subprocess
import copy
import numpy as np
import pandas as pd
import glob

from os import remove
from re import findall
from random import shuffle,choice
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import euclidean
from itertools import combinations
from math import sqrt
from scipy.spatial import ConvexHull

from ase.atoms import symbols2numbers
from ase.neighborlist import NeighborList
from ase.utils import basestring
from ase.cluster.factory import GCD
from ase.visualize import view
from ase.io import write,read
from ase.data import chemical_symbols
from ase.spacegroup import Spacegroup

from pymatgen.analysis.wulff import WulffShape

nonMetals = ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
                  'Si', 'P', 'S', 'Cl', 'Ar',
                  'Ge', 'As', 'Se', 'Br', 'Kr',
                  'Sb', 'Te', 'I', 'Xe',
                  'Po', 'At', 'Rn']
nonMetalsNumbers=symbols2numbers(nonMetals)

delta = 1e-10
_debug = False


def bcn_wulff_construction(symbol, surfaces, energies, size, structure,
                       rounding='closest', latticeconstant=None, 
                       debug=False, maxiter=100,center=[0.,0.,0.],stoichiometryMethod=1,np0=False):
    """Create a cluster using the Wulff construction.

    A cluster is created with approximately the number of atoms
    specified, following the Wulff construction, i.e. minimizing the
    surface energy of the cluster.

    Parameters:
    -----------

    symbol: atom object.

    surfaces: A list of surfaces. Each surface is an (h, k, l) tuple or
    list of integers.

    energies: A list of surface energies for the surfaces.

    size: The desired aproximate size.

    structure: The desired crystal structure.  Either one of the strings
    "fcc", "bcc", "sc", "hcp", "graphite"; or one of the cluster factory
    objects from the ase.cluster.XXX modules.

    rounding (optional): Specifies what should be done if no Wulff
    construction corresponds to exactly the requested size.
    Should be a string, either "above", "below" or "closest" (the
    default), meaning that the nearest cluster above or below - or the
    closest one - is created instead.

    latticeconstant (optional): The lattice constant.  If not given,
    extracted from ase.data.

    debug (optional): If non-zero, information about the iteration towards
    the right cluster size is printed.

    center: The origin of coordinates

    stoichiometryMethod: Method to transform Np0 in Np stoichometric 0 Bruno, 1 Danilo

    np0: Only gets the Np0, by means, the one that is build by plane replication
    """
    global _debug
    _debug = debug

    if debug:
        if type(size) == float:
            print('Wulff: Aiming for cluster with radius %i Angstrom (%s)' %
                  (size, rounding))
        elif type(size) == int:
            print('Wulff: Aiming for cluster with %i atoms (%s)' %
                  (size, rounding))

        if rounding not in ['above', 'below', 'closest']:
            raise ValueError('Invalid rounding: %s' % rounding)

    # Interpret structure, if it is a string.
    if isinstance(structure, basestring):
        if structure == 'fcc':
            ##STRUCTURE INHERITS FROM CLASSES IN FACTORY
            from ase.cluster.cubic import FaceCenteredCubic as structure
        elif structure == 'bcc':
            from ase.cluster.cubic import BodyCenteredCubic as structure
        elif structure == 'sc':
            from ase.cluster.cubic import SimpleCubic as structure
        elif structure == 'hcp':
            from ase.cluster.hexagonal import HexagonalClosedPacked as structure
        elif structure == 'graphite':
            from ase.cluster.hexagonal import Graphite as structure
        elif structure == 'ext':
            from bcn_cut_cluster import CutCluster as structure
        else:
            error = 'Crystal structure %s is not supported.' % structure
            raise NotImplementedError(error)

    # Check number of surfaces
    nsurf = len(surfaces)
    if len(energies) != nsurf:
        raise ValueError('The energies array should contain %d values.'
                         % (nsurf,))

    #Calculate the interplanar distance
    recCell=symbol.get_reciprocal_cell()
    dArray=interplanarDistance(recCell,surfaces)

    # Get the equivalent surfaces
    eq=equivalentSurfaces(symbol,surfaces)
    #Calculate the normal normalized vectors for each surface
    norms=planesNorms(eq,recCell)
    # Get the ideal wulffPercentages
    ideal_wulff_fractions=idealWulffFractions(symbol,surfaces,energies)


    if type(size) == float:
        """This is the loop to get the NP closest to the desired size"""
        if len(energies) == 1:
            scale_f = np.array([0.5])
            distances = scale_f*size
            layers = np.array([distances/d])
            atoms_midpoint = make_atoms_dist(symbol, surfaces, layers, distances, 
                                structure, center, latticeconstant)
            small = large = None
            
            if np.mean(atoms_midpoint.get_cell_lengths_and_angles()[0:3]) == size:
                midpoint = scale_f
            else:
                small = large = scale_f
                if np.mean(atoms_midpoint.get_cell_lengths_and_angles()[0:3]) > size:
                    large = scale_f
                    while np.mean(atoms_midpoint.get_cell_lengths_and_angles()[0:3]) > size:
                        scale_f = scale_f/2.
                        distances = scale_f*size
                        layers = np.array([distances/d])
                        atoms_midpoint = make_atoms_dist(symbol, surfaces, layers, distances, 
                                structure, center, latticeconstant)
                    small = scale_f
                    midpoint = small
                elif np.mean(atoms_midpoint.get_cell_lengths_and_angles()[0:3]) < size:
                    small = scale_f
                    while np.mean(atoms_midpoint.get_cell_lengths_and_angles()[0:3]) < size:
                        scale_f = scale_f*2.
                        distances = scale_f*size
                        layers = np.array([distances/d])
                        atoms_midpoint = make_atoms_dist(symbol, surfaces, layers, distances, 
                                structure, center, latticeconstant)
                    large = scale_f
                    midpoint = large
        else:
            # print('dArray\n',dArray)

            small = np.array(energies)/((max(energies)*2.))
            large = np.array(energies)/((min(energies)*2.))
            midpoint = (large+small)/2.
            distances = midpoint*size
            layers= distances/dArray
            # print('layers\n',layers)
            # print('size\n',size)
            atoms_midpoint = make_atoms_dist(symbol, surfaces, layers, distances, 
                                        structure, center, latticeconstant)
            
            # print("Initial NP",atoms_midpoint.get_chemical_formula())
            # view(atoms_midpoint)
            name = atoms_midpoint.get_chemical_formula()+str(center)+"_NP0.xyz"
            write(name,atoms_midpoint,format='xyz',columns=['symbols', 'positions'])

            #Evaluating the np0
            np0Properties=[]

            
            minCoord=check_min_coord(atoms_midpoint)
            areasIndex=areaCalculation(atoms_midpoint,norms)
            # print('areasIndex',areasIndex)
            plane_area=planeArea(symbol,areasIndex,surfaces)
            # print('plane_area',plane_area)
            # print('--------------')
            # print(len(plane_area))

            # Calculate the Wulff-like index
            wulff_like=wulffLike(symbol,ideal_wulff_fractions,plane_area[1])
            # print(wulff_like)

            # view(atoms_midpoint)
        
            # """
            # For now I will keep it here too
            # """
            # name = atoms_midpoint.get_chemical_formula()+str(center)+"_NP0.xyz"
            # write(name,atoms_midpoint,format='xyz',columns=['symbols', 'positions'])
            # """
            # testing it the np0 contains metal atoms with lower coordination than the half of the maximum coordination
            # """
            # if check_min_coord(atoms_midpoint)==True:
            #     print('The initial NP contain metals with coordination lower than the half of the maximum coordination')
            #     return none
            #     # raise systemexit(0)

            # if option == 0:
            #     if all(np.sort(symbol.get_all_distances())[:,1]-max(np.sort(symbol.get_all_distances())[:,1]) < 0.2):
            #         n_neighbour = max(np.sort(symbol.get_all_distances())[:,1])
            #     else:
            #         n_neighbour = none
            #     coordination(atoms_midpoint,debug,size,n_neighbour)
            #     os.chdir('../')
            #     return atoms_midpoint
            if np0==True:
                np0Properties=[atoms_midpoint.get_chemical_formula()]
                np0Properties.extend(minCoord)
                np0Properties.append(plane_area[0])
                np0Properties.extend(wulff_like)
                # name = atoms_midpoint.get_chemical_formula()+str(center)+"_NP0.xyz"
                # write(name,atoms_midpoint,format='xyz',columns=['symbols', 'positions'])
                return np0Properties
            else:
                # view(atoms_midpoint) 
                reduceNano(symbol,atoms_midpoint,size)
                # os.chdir('../')
def make_atoms_dist(symbol, surfaces, layers, distances, structure, center, latticeconstant):
    # print("here")
    layers = np.round(layers).astype(int)
    # print("1layers",layers)
    atoms = structure(symbol, surfaces, layers, distances, center= center,                   
                      latticeconstant=latticeconstant,debug=1)

    return (atoms)

def coordination(atoms,debug,size,n_neighbour):
    #time_0_coord = time.time()
    """Now find how many atoms have the first coordination shell
    """
    
    newpath = './{}'.format(str(int(size)))
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(newpath)
    name=atoms.get_chemical_formula()+'_NP0.xyz'
    write(name,atoms,format='xyz',columns=['symbols', 'positions'])

    nearest_neighbour= []
    if n_neighbour is None:
        for i in range(len(atoms.get_atomic_numbers())): 
            nearest_neighbour.append(np.min([x for x in atoms.get_all_distances()[i] if x>0]))
        nearest_neighbour_av = np.average(nearest_neighbour)
        for i in nearest_neighbour:
            if i > nearest_neighbour_av*1.5:
                print("EXITING: there is something strange with the distances, check NP0 for size", int(size))
                return None

    else:
        nearest_neighbour = [n_neighbour]*len(atoms.get_atomic_numbers())


    final = False
    while final == False:
        final = True
        
        C = make_C(atoms,nearest_neighbour)
        # atomicNumbers=atoms.get_atomic_numbers()
        # atomsCoordination=zip(atomicNumbers,C)

        # # for i in atomsCoordination:
        # #     if 
        
        coord=np.empty((0,5))
        for d in set(atoms.get_atomic_numbers()):
            a=np.array([d, np.mean([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                        np.max([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                        np.min([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                        chemical_symbols[d]])
            coord = np.append(coord,[a],axis=0)
        coord = coord[coord[:,4].argsort()]
        print("coord \n",coord)
        
        if check_stoich(atoms,coord) is 'stop':
            print("Exiting because the structure is nonmetal defective")
            return None

        E=[]
        for i in atoms.get_atomic_numbers():
            if int(i) in nonMetalsNumbers:
                E.append(1)
            else:
                for n in coord:
                    if i == int(n[0]):
                        E.append((int(n[2]))/2)
        # print('E',E)
        D = []
        print('atoms pre pop\n',atoms.get_chemical_formula())
        for i,j in enumerate(C):
            if j < E[i]:
                D.append(i)
        for i,j in enumerate(D):
            atoms.pop(j-i)
            nearest_neighbour.pop(j-i)
            C = np.delete(C,j-i)
        print('atoms post pop\n',atoms.get_chemical_formula())
        check_stoich(atoms)
        
        atoms_only_metal = copy.deepcopy(atoms)

        del atoms_only_metal[[atom.index for atom in atoms if atom.symbol in nonMetals ]]
        
        # del atoms_only_metal[[atom.index for atom in atoms if atom.symbol=='O']]
        # del atoms_only_metal[[atom.index for atom in atoms if atom.symbol=='S']]
        center_of_metal = atoms_only_metal.get_center_of_mass()
        
        S = None
        
        dev_s = 100000000.0
        index=0
        if debug == 1:
            print("Atoms to be removed",atoms.excess)
        if atoms.excess is not None:
            """
            atoms.excess is an atribute vector that contains the atom excess to be 
            stoichiometric. e.g. a (IrO2)33 initial nano contains 33 Ir and 80 O atoms
            so to be stoichiometric we have to remove 18 O, this is representated
            by the vector atom.excess [0 14].
            From here until final the stuff is:
            Calculate coordination aka C=make_C
            Generate the an array of shufled list aka S=make_F
            For every element of S recover j non metal atoms.
            in this case 14 metal atoms, the first 14 in tmp_l.

            Kepping in mind that eliminating atoms change the numeration order
            is mandatory to include the changes, this is done by tot.
            in the first elimination, tot =0, so eliminate the atom k
            in the second cycle, tot=1, so eliminate the k-1 atom, that 
            is equivalent to the k atom in the initial atoms object.

            """
            print('int(coord[i,0])',int(coord[0,0]))
            # time.sleep(10)
            for i,j in enumerate(atoms.excess):
                # print(i,j,'i,j')
                if j > 0:
                    C = make_C(atoms,nearest_neighbour)
                    S = make_F(atoms,C,nearest_neighbour,debug)
                    # print('s\n',S)
                    # S =make_F_Test(atoms,coordination_testing)
                    # break
                    for h in range(len(S)):
                        atoms1 = copy.deepcopy(atoms)
                        C1 = copy.deepcopy(C)
                        E1 = copy.deepcopy(E)
                        nearest_neighbour1 = copy.deepcopy(nearest_neighbour)
                        ind=0
                        tmp_l = []
                        for m in S[h]:
                            """
                            This part could be made more pythonic
                            """
                            if ind < int(j):
                                if atoms1.get_atomic_numbers()[m] == int(coord[i,0]):
                                    tmp_l.append(m)
                                    ind = ind + 1
                        # print('tmp_l',tmp_l)
                        tot = 0
                        for k in sorted(tmp_l):
                            # print('tot',tot)
                            atoms1.pop(k-tot)
                            nearest_neighbour1.pop(k-tot)
                            
                            C1 = np.delete(C1,k-tot)
                            E1 = np.delete(E1,k-tot)
                            tot += 1
                        # time.sleep(10)
                        atoms_only_oxygen = copy.deepcopy(atoms1)
                        del atoms_only_oxygen[[atom.index for atom in atoms1 if atom.symbol not in nonMetals]]
                        center_of_oxygen = atoms_only_oxygen.get_center_of_mass()
                        dev = np.std(abs(center_of_metal-atoms_only_oxygen.get_center_of_mass()))
                        dev_p = float("{:.7f}".format(round(float(dev*100),7)))
                        """
                        THIS WRITE IS FOR TESTING PURPOSES
                        
                        if dev_p == 0.0:
                            index += 1
                            name = atoms1.get_chemical_formula()+'_NPtest'+str(index)+".xyz"
                            write(name,atoms1,format='xyz',columns=['symbols', 'positions'])
                        """
                        if debug == True:
                            # comment = ("DC = "+str(dev*100)+' Lattice="' +
                            #            ' '.join([str(x) for x in np.reshape(atoms.cell.T,
                            #                             9, order='F')]) +
                            #            '"')
                            comment = ("DC = "+str(dev*100))
                            name = atoms1.get_chemical_formula()+'_'+str(dev_p)+".xyz"
                            write(name,atoms1,format='xyz',comment=comment,columns=['symbols', 'positions'])
                        if dev < dev_s:
                            dev_s = dev
                            atoms_f = copy.deepcopy(atoms1) 
                            nearest_neighbour_f = copy.deepcopy(nearest_neighbour1)
                            C_f = copy.deepcopy(C1)
                            E_f = copy.deepcopy(E1)
                            if debug == False:
                                if round(dev_s,7) == 0.:
                                    break
            atoms = copy.deepcopy(atoms_f)
            nearest_neighbour = copy.deepcopy(nearest_neighbour_f)
            C = copy.deepcopy(C_f)
            E = copy.deepcopy(E_f)

        C = make_C(atoms,nearest_neighbour)

        for i in range(len(atoms.get_atomic_numbers())):
            if C[i] < E[i]:
                final = False
                print("final",final)
                break
          

    coord_final=np.empty((0,2))
    for d in set(atoms.get_atomic_numbers()):
        a=np.array([d, np.mean([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d])])
        coord_final = np.append(coord_final,[a],axis=0)
    
    check_stoich(atoms)
    if atoms.stoichiometry == True:
        
        if S == None:
            if debug == 1:
                print("It did't go into the coordination loop, check why")
            atoms_only_oxygen = copy.deepcopy(atoms)
            del atoms_only_oxygen[[atom.index for atom in atoms if atom.symbol not in nonMetals]]
            # del atoms_only_oxygen[[atom.index for atom in atoms if atom.symbol!='O']]
            center_of_oxygen = atoms_only_oxygen.get_center_of_mass()
            
            dev_s = np.std(abs(center_of_metal-center_of_oxygen))
            dev_p = float("{:.7f}".format(round(float(dev_s*100),7)))

        else:
            if debug == 1:
                print(len(S)," different combinations were tried resulting in", 
                      len([name for name in os.listdir('.') if os.path.isfile(name)])-1,"final NPs")
                """
                Identify the equal models with sprint coordinates
                """
                # print (os.listdir('.'))
                singulizator(glob.glob('*.xyz'))


        dev_p = float("{:.7f}".format(round(float(dev_s*100),7)))
        name = atoms.get_chemical_formula()+'_NPf_'+str(dev_p)+".xyz"
        # comment = ("DC = "+ str(dev_s*100) +
        #        str(np.reshape(coord_final,(1,4))) +
        #        ' Lattice="' +
        #        'a= '+str(atoms.cell[0,0])+
        #        ' b= '+str(atoms.cell[1,1])+
        #        ' c= '+str(atoms.cell[2,2]) +
        #        '"')
        comment = ("DC = "+ str(dev_s*100)) 

        print("Final NP", atoms.get_chemical_formula(), "| DC =", dev_p, "| coord", coord_final[:,0], coord_final[:,1])
        write(name,atoms,format='xyz',comment=comment,columns=['symbols', 'positions'])
        return atoms
    

def check_stoich(atoms,coord=None):
    stoich_unit = np.array(findall('\d+',atoms.unit_cell_formula))
    stoich_unit = stoich_unit.astype(np.int)
    divisor_unit = GCD(stoich_unit[0],stoich_unit[1])
    unit_formula_unit = stoich_unit/divisor_unit
    stoich_cl = np.array(findall('\d+',atoms.get_chemical_formula()))
    stoich_cl = stoich_cl.astype(np.int)
    a = 0
    b = 0
    if coord is not None:
        for i in range(len(stoich_cl)):
            # if coord[i][4] == 'S' or coord[i][4] == 'O':
            if coord[i][4] in nonMetals:
                a = stoich_cl[i]/unit_formula_unit[i]
            else:
                b = stoich_cl[i]/unit_formula_unit[i]
        if (b == 0) or (a == 0):
            return 'stop'
        if b > a:
            return 'stop'
    atoms.excess = None
    if stoich_cl.size != unit_formula_unit.size:
        atoms.stoichiometry = False
    else:
        divisor_cl = GCD(stoich_cl[0],stoich_cl[1])
        unit_formula_cluster = stoich_cl/divisor_cl
        if sum(abs((unit_formula_cluster - unit_formula_unit))) == 0:
            atoms.stoichiometry = True
        else:
            atoms.stoichiometry = False
            t= np.argmin((stoich_cl/np.amin(stoich_cl))/unit_formula_unit)
            ideal_cl = stoich_cl[t]//unit_formula_unit[t] * unit_formula_unit
            atoms.excess = stoich_cl - ideal_cl
  
def make_C(atoms,nearest_neighbour):
    C=[]
    """Let's test this cause I am not 100% sure"""
    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)
    for i in range(len(atoms.get_atomic_numbers())):
        indices, offsets = nl.get_neighbors(i)
        C.append(len(indices))
    # print('C\n',C)
    return C
    

def make_F(atoms,C,nearest_neighbour,debug):
    #time_0_make_F = time.time()
    """
    Create the F vector, which contains the index of the atom to which the 
    singly coordinated oxygen atoms are linked to.
    The other values are set =len(C) for the other oxygen atoms and
    =len(C)+1 for the other atom types
    """
    # print('c',C)
    # for i,j in enumerate(C):
        # print('element,index, coordination ',atoms[i].symbol,i,j)
    if debug == 1:
        print("Start calculating combinations")
        time_F0 = time.time()
    # F_test=make_F_Test(atoms)
    # print('F_test\n',F_test)

    F=[]
    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)
    for i in range(len(C)):
            if atoms.get_atomic_numbers()[i] in nonMetalsNumbers:
                # print(atoms.get_atomic_numbers()[i],i,'holi')
            # if atoms.get_atomic_numbers()[i] == (8 or 16):
                if C[i] == 1:
                    # print('i',i)
                    indices, offsets = nl.get_neighbors(i)
                    # print(indices,i,'indices,i')
                    if len(indices) == 1:
                        F.append(indices[0])
                        # print(i,indices[0],'join')
                    else:
                        print("There is a problema....",indices,C[i])
                        # exit()
                else:

                    F.append(len(C))
            else:
                F.append(len(C)+1)
    # print('aquiiii')
    # print('len(F)\n',len(F))
    # print('F\n',F)
    """
    A nex coordination list is created by adding the values 10 and 11
    for the higher coordination oxygen atoms and other atom types, respectively.
    This is to avoid having an error in the following loop, where the coordination
    is checked and the values len(C) and len(C)+1 would be out of bound
    """
    c = list(C)
    c.append(11)
    c.append(12)
    # print('len(c)\n',len(c),'\n',c)

    K=[]
    n_tests = 1000
    for i in range(n_tests):
        K.append([])
        # break
    # print ('K\n',K)
    """
    l=list of allowed coordinations

    NOT SURE THIS IS STILL VALID
    In the first part of the loop I generate the following lists:
    g: it is a list of indices grouping together the atoms
    linked to singly coordinated atoms, which have the same coordination
    a: gives a list of indices of the order to pick from the atoms.whatever
    in order to select the atoms in g
    In the second loop the G and A lists are equivalent to the g and a lists,
    but the indices g all appear once before appearing a second time. This way
    first all O atoms linked to 6 coordinated atoms, once these lost one O
    they are 5 coordinated atoms and should go after.
    """
    l = [10,9,8,7,6,5,4,3,2,1,11,12]

    F_safe = copy.deepcopy(F)
    c_safe = copy.deepcopy(c)
    for y in range(n_tests):
        a=[]
        F = copy.deepcopy(F_safe)
        c = copy.deepcopy(c_safe)
        for j in l:
            # print('j ',j)
            g = []
            r = list(range(len(F)))
            # print(len(r),'lenr')
            shuffle(r)
            # print(r)
            for i in r:
                if F[i] != None:
                    if c[F[i]] == j:
                        # print('c[F[i]]',c[F[i]],i)
                        # time.sleep(3)
                        if j < 11:
                            if F[i] not in g:
                                g.append(F[i])
                                a.append(i)
                                F[i] = None
                            else:
                                c[F[i]] = j-1
                        else:
                            g.append(F[i])
                            a.append(i)
                            F[i] = None
                # print (i,j,c[F[i]])
                # time.sleep(5)
        # print('a',a)
        # print('g',g)
        K[y].extend(a)

    if debug == 1:
        time_F1 = time.time()
        print("Total time to calculate combinations", round(time_F1-time_F0,5)," s")
    return K
def check_min_coord(atoms):
    """function that identify if the nanoparticle not contain lower coordinated metals
    args: atoms
    return: characterization([metals,nonmetals,undercoordinated,globalCoord,])
            list of information of np0.

    """
    nearest_neighbour= []
    indexes=[]
    characterization=[]

    for i in range(len(atoms.get_atomic_numbers())):
        nearest_neighbour.append(np.min([x for x in atoms.get_all_distances()[i] if x>0]))

    # print (nearest_neighbour)

    C=make_C(atoms,nearest_neighbour)
    atomIndex=[atom.index for atom in atoms]
    for i in atomIndex:
        indexes.append([i,C[i]])

    ##Sum the coordination of all elements
    globalCoord=np.sum(indexes,axis=0)[1]

    # Get the metals
    metalAtom=[atom.index for atom in atoms if atom.symbol not in nonMetals]
    #Calculate the nonMetals as the diference between metals and total atoms
    nonMetalsNumber=len(atoms)-len(metalAtom)

    #Get the metals coordinations
    metalsCoordinations=[i[1] for i in indexes if i[0] in metalAtom]
    # print(metalsCoordinations)/

    maxCoord=np.amax(metalsCoordinations)
    # print('maxCoord:',maxCoord)

    ##Filling characterization list

    characterization.append(len(metalAtom))
    characterization.append(nonMetalsNumber)
    #Evaluate if metals have coordination larger than
    #the half of maximium coordination
    minCoord=np.min(metalsCoordinations)
    # print('minCoord',minCoord)
    if minCoord>=maxCoord/2:
        coordTest=True
    else:
        coordTest=False

    # coordTest=all(i >= maxCoord/2 for i in metalsCoordinations)
    # if coordTest==False:
        # print(metalsCoordinations) 
    characterization.append(coordTest)

    characterization.append(globalCoord)

    return characterization

def singulizator(nanoList):
    """
    Function that eliminates the nanoparticles
    that are equivalent by SPRINT coordinates
    """

    print('Enter in the singulizator')
    time_F0 = time.time()

    sprintCoordinates=[]
    results=[]
    for i in nanoList:
        # print (i)
        sprintCoordinates.append(sprint(i))
        # break
    for c in combinations(range(len(sprintCoordinates)),2):
    #     # print (c[0],c[1],'c')
        if compare(sprintCoordinates[c[0]],sprintCoordinates[c[1]]) ==True:
            results.append(c)

    # print(results)
    """
    keeping in mind that a repeated structure can appear
    on both columns, I just take the first
    """
    # for i in results:
    #     print('NP '+nanoList[i[0]]+' and '+nanoList[i[1]]+ ' are equal')

    
    results1=[i[0] for i in results]
    # print (results1)
    toRemove=list(set(results1))

    for i in toRemove:
        # print(i)
        # print('NP '+nanoList[results[i][0]]+' and '+nanoList[results[i][1]]+ ' are equal')
        # print('Removing '+nanoList[results[i][0]])
        remove(nanoList[i])
    finalModels=len(nanoList)-len(toRemove)
    print('Removed NPs:',len(toRemove))
    print('Final models:',finalModels)

    time_F1 = time.time()
    print("Total time singulizator", round(time_F1-time_F0,5)," s\n")


def sprint(nano):
    """
    Calculate the sprint coordinates matrix for a nanoparticle.

    First calculate the coordination, then build the adjacency
    matrix. To calculate the coordination firstly generates a 
    nearest_neighbour cutoffs for NeighborList.

    The C list contains the atom and the binding atoms indices.
    From C list we build the adjMatrix. The idea is translate from
    coordination list to adjacency matrix.

    Then, calculate the sprint coordinates
    """
    atoms=read(nano,format='xyz')
    atoms.center(vacuum=20)
    adjacencyName=nano+'dat'

    # print (nano)
    nearest_neighbour=[]
    C=[]

    for i in range(len(atoms.get_atomic_numbers())):
        nearest_neighbour.append(np.min([x for x in atoms.get_all_distances()[i] if x>0]))

    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)

    for i in range(len(atoms.get_atomic_numbers())):
        indices, offsets = nl.get_neighbors(i)
        C.append([i,indices])
        # print(i,indices) 

    m=len(C)
    adjMatrix=np.zeros((m,m))
    for i in C:
        for j in i[1]:
            adjMatrix[i[0],j]=1.0
    # np.savetxt('adjMatrix',adjMatrix,newline='\n',fmt='%.1f')

    """
    Diagonal elements defined by 1+zi/10 if i is non metal
    and 1+zi/100 if is metal
    """
    numbers=symbols2numbers(atoms.get_atomic_numbers())
    # print(numbers)
    for i in range(len(adjMatrix)):
        if numbers[i] <=99 :
            adjMatrix[i][i]=1+float(numbers[i])/10
        else:
            adjMatrix[i][i]=1+float(numbers[i])/100

    # np.savetxt(adjacencyName,adjMatrix,newline='\n',fmt='%.3f')

    # Calculating the largest algebraic eigenvalues and their 
    # correspondent eigenvector
    val,vec=eigsh(adjMatrix,k=1,which='LA')

    # print(val,'val')
    # print(vec)
    
    # Sorting and using positive eigenvector values (by definition)
    # to calculate the sprint coordinates

    vecAbs=[abs(i)for i in vec]
    vecAbsSort=sorted(vecAbs)
    s=[sqrt(len(adjMatrix))*val[0]*i[0] for i in vecAbsSort]
    sFormated=['{:.3f}'.format(i) for i in s]
    # print (s)
    # print(sFormated)
    # print (len(s))
    return sFormated

def compare(sprint0,sprint1):
    """
    compare the SPRINT coordinates between two nanoparticles.
    If two NP has the same sprint coordinates, both are equally
    connected.
    """
    # print(sprint0,'\n',sprint1) 
    # diff=(list(set(sprint0) - set(sprint1)))
    if len(sprint0)==len(sprint1):
        diff=(list(set(sprint0) - set(sprint1)))
        if len(diff)==0:
            return True
def coordination_testing(atoms):
    print('entre a coordination_testing\n')
    atoms.center(vacuum=10)
    nearest_neighbour=[]
    for i in range(len(atoms)):
        nearest_neighbour.append(np.min([x for x in atoms.get_all_distances()[i] if x>0]))


    C=[]
    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)
    for i in range(len(atoms.get_atomic_numbers())):
        indices, offsets = nl.get_neighbors(i)
        C.append(len(indices))
    return C
def reduceNano(symbol,atoms,size):
    """
    function that make the nano stoichiometrically
    """
    print('Enter to reduceNano')
    time_F0 = time.time()
    check_stoich(atoms)
    
    # newpath = './{}'.format(str(int(size)))
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)
    # os.chdir(newpath)

    name=atoms.get_chemical_formula()+'_NP0.xyz'
    write(name,atoms,format='xyz',columns=['symbols', 'positions'])

    """
    Check the nano 0 quality
    """
    nearest_neighbour=[]
    for i in range(len(atoms)):
        nearest_neighbour.append(np.min([x for x in atoms.get_all_distances()[i] if x>0]))

    C = make_C(atoms,nearest_neighbour)
    
    coord=np.empty((0,5))
    for d in set(atoms.get_atomic_numbers()):
        a=np.array([d, np.mean([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                    np.max([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                    np.min([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                    chemical_symbols[d]])
        coord = np.append(coord,[a],axis=0)
    coord = coord[coord[:,4].argsort()]
    # print("coord \n",coord)
    
    if check_stoich(atoms,coord) is 'stop':
        print("Exiting because the structure is nonmetal defective")
        return None

    """
    If the nano pass the test, the program can advance.
    """
    # print(atoms.excess)
    # if atoms.excess.any==None:
    #     print('is stoichiometric')
    #     return None
    # else:
    for i in atoms.excess:
        if i !=0:
            excess=i
    # print(excess)

    C=[]
    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)
    for i in range(len(atoms.get_atomic_numbers())):
        indices, offsets = nl.get_neighbors(i)
        C.append([i,indices])

    '''
    4 lists:
    singly: contains the singly coordinated atoms
    father: contains the heavy metal atoms which singly
    coordinated atoms are bounded
    coordFather that is the coordination of bounded
    fatherFull that is the combination of father and their coordination.
    coordFather:contains the coordination of bounded
    fatherFull: contains the combination of father and their coordination.
    '''
    
    singly=[i for i in range(len(atoms)) if len(C[i][1])==1]
    singly_bak=copy.deepcopy(singly)

    father=list(set([C[i][1][0] for i in singly]))
    father_bak=copy.deepcopy(father)

    coordFather=[len(C[i][1]) for i in father]
    coordFather_bak=copy.deepcopy(coordFather)

    fatherFull=[[i,coordFather[n]] for n,i in enumerate(father)]

    fatherFull_bak=copy.deepcopy(fatherFull)

    """
    allowedCoordination must to be generalized
    the upper limit is half of maximum coordination -1
    and the inferior limit is the maximum
    coordination. i.e. for fluorite, the maximum coordination
    is 8, so using list(range(8,3,-1)) we obtain the list
    [8, 7, 6, 5, 4] that is fully functional.
    """
    maxCord=int(np.max(coordFather))
    # print (maxCord)
    mid=int(0.5*maxCord-1)

    allowedCoordination=list(range(maxCord,mid,-1))
    # print('allowedCoordinations')
    # print(allowedCoordination)


    replicas=1000
    """
    To have a large amounth of conformation we generate
    1000 replicas for removing atoms. 
    To make the selection random we use shuffle and 
    choice. 
    The loop basically select the metal
    atom of higest coordination,aka father, identify the singly coordinated 
    atoms bounded to it and chose one randomly.
    Then append the selected and reduce the coordination of father.
    the process is repeated until the len of remove are equal to 
    excess.

    """
    S=[]

    for r in range(replicas):
        toRemove=[]
        fatherFull=copy.deepcopy(fatherFull_bak)
        singly=copy.deepcopy(singly_bak)
        for i in allowedCoordination:
            shuffle(fatherFull)
            # print('fatherFull, evaluated coordination',fatherFull,i)
            for n,j in enumerate(fatherFull):
                if fatherFull[n][1]==i:
                    # print('fatherFull[n][1]',fatherFull[n])
                    singlyFather=[k for k in C[j[0]][1] if k in singly]
                    if len(singlyFather)>0:
                        # print('singlyFather',singlyFather)
                        chosen=choice(singlyFather)
                        # print('chosen',chosen)
                        if chosen not in toRemove:
                            if len(toRemove)==excess:
                                break
                            toRemove.append(chosen)
                            # print('singly',singly)
                            singly.remove(chosen)
                            fatherFull[n][1]=fatherFull[n][1]-1
            # print('len(toRemove)',len(toRemove))
        # print(len(toRemove))
        S.append(sorted(toRemove))

    """
    at the end we get an array S with 10000 list of atoms
    to be removed. Previous to the removal and to make the things faster
    we remove duplicates (I guess that is not duplicates in the list)
    """
    nanoList=[]

    '''
    Generate the list of pairs and select the repeated pairs
    the aceptance criteria is if the intersection between
    two s are iqual to the len of the first s. 
    The repeated list is set and reversed before remove 
    s elements 
    '''
    pairs=[c for c in combinations(range(1000),2)]

    repeatedS=[]
    for c in pairs:
        # print (c[0],S[c[0]])

        if len(list(set(S[c[0]]).intersection(S[c[1]]))) == len(S[c[0]]):
            # print(c,' are duplicates')
            repeatedS.append(c[0])
            # del S[c[0]]
    # print (n)

    uniqueRepeated=list(set(repeatedS))
    uniqueRepeated.sort(reverse=True)

    for i in uniqueRepeated:
        del S[i]
    """
    Build the nanoparticles removing the s atom list. Then, calculate the DC
    """
    atomsOnlyMetal=copy.deepcopy(atoms)
    del atomsOnlyMetal[[atom.index for atom in atomsOnlyMetal if atom.symbol in nonMetals]]
    centerOfMetal = atomsOnlyMetal.get_center_of_mass()
    # print('centerOfMetal',centerOfMetal)
    print('stoichiometric NPs:',len(S))
    for n,s in enumerate(S):
        NP=copy.deepcopy(atoms)
        s.sort(reverse=True)
        del NP[[s]]

        #DC calculation
        atomsOnlyNotMetal = copy.deepcopy(NP)
        del atomsOnlyNotMetal[[atom.index for atom in atomsOnlyNotMetal if atom.symbol not in nonMetals]]
        centerOfNonMetal = atomsOnlyNotMetal.get_center_of_mass()
        # print('centerOfNotMetal',centerOfNonMetal)
        dev = np.std(abs(centerOfMetal-centerOfNonMetal))
        dev_p = float("{:.7f}".format(round(float(dev*100),7)))
        name=str(NP.get_chemical_formula(mode='hill'))+'_'+str(dev_p)+'_'+str(n)+'.xyz'
        # print('name',name)
        nanoList.append(name)
        #Saving NP
        write(name,NP,format='xyz')
        #calculating coulomb energy
        #calculating real dipole moment
        coulomb_energy=coulombEnergy(symbol,NP)
        # print('coulomb_energy',coulomb_energy)
        dipole_moment=dipole(NP)
        comment='E:'+str(coulomb_energy)+',mu:'+str(dipole_moment)
        #replace the ase standard comment by ours
        command='sed -i \'2s/.*/'+comment+'/\' '+name
        # print(command)
        subprocess.run(command,shell=True)
        # view(NP)
        # break

    time_F1 = time.time()
    print("Total time reduceNano", round(time_F1-time_F0,5)," s\n")

    #Calling the singulizator function 
    singulizator(nanoList)

def interplanarDistance(recCell,millerIndexes): 
    """Function that calculates the interplanar distances
    using 1/d_hkl^2 = hkl .dot. Gstar .dot. hkl equation.
    A Journey into Reciprocal Space: A Crystallographer's Perspective
    2-7
    Args:
        recCell(list): reciprocal cell of crystal structure
        millerIndexes(list): miller indexes of relevant surfaces
    Returns:
        distances(list): interplanar distance
    """
    # print(type(recCell))
    G=recCell
    # print(G)
    Gstar = np.dot(G, G.T)
    # print(Gstar)
    d=[]
    for indexes in millerIndexes:
        id2 = np.dot(indexes, np.dot(Gstar, indexes))
        d.append(np.sqrt(1/id2))
    # for n,i in enumerate(d):
    #     print(millerIndexes[n],d[n])

    return(d)
def equivalentSurfaces(atoms,millerIndexes):
    """Function that get the equivalent surfaces for a set of  millerIndexes
    Args:
        millerIndexes(list([])): list of miller Indexes
    Return:
        equivalentSurfaces(list([])): list of all equivalent miller indexes
    """
    surfaces = np.array(millerIndexes)
    sg=Spacegroup((int(str(atoms.info['spacegroup'])[0:3])))
    equivalent_surfaces=[]
    for s in millerIndexes:
        equivalent_surfaces.extend(sg.equivalent_reflections(s))

    return equivalent_surfaces

def planesNorms(millerIndexes,recCell):
    """Function that calculates the normalized
    normal vector of the miller indexes  
    Args:miller indexes(list)
    Return:norm(list)
    """
    norms=[]
    for millerIndex in millerIndexes:
        normal=np.dot(millerIndex,recCell)
        normal /=np.linalg.norm(normal)
        norms.append([millerIndex,normal])
        # norms[millerIndex]=normal
    # print(norms)
    return norms

def areaCalculation(atoms,norms):
    """Function that calculates the real areas of the faces
    of Np0 and their percentages.
    Args:
        atoms(Atoms): atoms object of NP0
        norms(List): list of norms of miller planes from
        the crystal
    Return:
        percentage(list): surfaces and their percentage
    
    Warning:
        Keep in mind that using the criteria of
    vector·normal=0  to asign the plane, you know
    that you can have two normal planes that are orthogonal
    to each vector, formaly axb, bxa.

    Solution:
        To ovecome the two normals problem,we have
        to evaluate if the normals are pointing
        inwards of outwards of the surface, 
        if the center of the simplex and the normal
        have the same direction, both are pointing
        outwards, else, inwards

    """
    # Steps:

    #     Use the convexHull to get the simplex

    #     evaluate if vectors that form the simplex are
    #     ortogonal to normal vectors of the miller index
    #     related plane.

    #     get the area of the simplexes and sum by plane

    #Reading the atoms object

    #Only get the metal positions, just to make it easy
    positions=np.array([atom.position[:] for atom in atoms if atom.symbol not in nonMetals])
    #Calculate the centroid
    centroid=positions.mean(axis=0)


    #Create the ConvexHull  object
    hull=ConvexHull(positions)
    # print(hull.area)

    #Identify the miller index of the simplices and calculate the area per miller index
    simplices=[]
    for n,simplex in enumerate(hull.simplices):
        u=positions[simplex[0]]-positions[simplex[1]]
        v=positions[simplex[1]]-positions[simplex[2]]
        area=np.abs(np.linalg.norm(np.cross(u, v)) / 2)
        # print(area)
        #Calculate the centroid of the simplex
        tempPos=np.asarray([positions[i] for i in simplex])
        simplexCentroid=tempPos.mean(axis=0)
        # print(tempPos)
        # print(simplexCentroid)
        #Calculate the normalized normal
        normalVector=np.cross(u, v)
        # print(normalVector)
        norm2=normalVector/np.linalg.norm(normalVector)
        # print(n2)
        #identify if is pointing inwards of outwards
        #1 change the origin of pmid to the centroid
        simplexCentroid2=simplexCentroid-centroid
        #2 Calculate the dot product between normalized normal and
        #simplexCentroid2 vector, if the dot product is positive, the normal
        #and the simplexCentroid2 have the same direction. By definition
        #the simplexCentroid2 is pointing outwards of centroid always.
        if np.dot(simplexCentroid2,normalVector)<0:
            norm2*=-1
        #3 compare the norms of each simplex with the crystal norms
        #The threshold of 1e-2 just has been tested for IrO2 and works well.
        #If the diference of between elements of the norm are less than threshold
        #evaluator is True, so that means that the simplex and the 
        #plane has the same normal so they are parallel.
        for i in norms:
            test=i[1]-norm2
            evaluator=all(np.abs(j)< 1e-2 for j in test)
            # print(evaluator)
            if evaluator==True:

                # print('normal of simplex',norm2)
                # print('normal of crystal',i[1])
                # print(np.sum(i[1]-norm2))
                # print (n,i[0])
                # print('--')

                simplices.append([str(i[0]),area])

    areasPerMiller=[]
    areasIndex=[]
    for i in norms:
        millerArea=0
        for j in simplices:
            if str(i[0])==str(j[0]):
                # print(j[0],i[0])
                # print('---')
                millerArea+=j[1]

        areasPerMiller.append(millerArea)
        percentage=millerArea/hull.area
        areasIndex.append([str(i[0]),percentage])
        # print(i[0],percentage)
    # print(areasIndex)
    return areasIndex

def planeArea(atoms,areasIndex,millerIndexes):
    """Function that get the areas per miller index.
    Args:
        atoms(Atoms): atoms object
        areasIndex([index,area]): list of indexes and areas per each index
        millerIndexes([millerIndex]): list of initial miller indexes
    Return:
        areasPerInitialIndex([index,area]): list of initial indexes and areas per each index

    """
    #get the spacegroup object
    surfaces = np.array(millerIndexes)
    sg=Spacegroup((int(str(atoms.info['spacegroup'])[0:3])))
    
    #For each miller index get the equivalent reflexions and append
    # as strings in equivalent_strings
    areasPerInitialIndex=[]
    symmetric=[]
    for s in surfaces:
        areaPerSurface=[]
        equivalentSurfaces=[]
        equivalentStrings=[]
        equivalentSurfaces.append(sg.equivalent_reflections(s))
        for indexes in equivalentSurfaces:
            for index in indexes:
                equivalentStrings.append(str(index))
        #Make a comparison between the index of area Index and the equivalent_strings lists
        #and accumulate the areas
        for area in areasIndex:
            if area[0] in equivalentStrings:
                areaPerSurface.append(area[1])
        # To evaluate if the nano faces are symmetrical
        # we compare the areas per normal. If the areas are not
        # equal we discard the model
        if np.sum(areaPerSurface)==0.0:
            # print(s,np.sum(areaPerSurface),'\n----')
            areasPerInitialIndex.append([s,np.sum(areaPerSurface)])
        else:
            areasPerInitialIndex.append([s,np.sum(areaPerSurface)])
            # print(s,np.sum(areaPerSurface),'\n----')
            # print(areaPerSurface)
            # print('-------------')
            tempArea=["%.4f"%i for i in areaPerSurface if i!=0.0]
            # print(s,tempArea)
            # print('-------------')
            # temp=np.asarray(tempArea)
            unique=list(set((tempArea)))
            if len(unique)>1:
                # print('not symmetric')
                symmetric.append(0)
            else:
                # print('symmetric')
                symmetric.append(1)
    if 0 in symmetric:
        # print('Non symmetric grow')
        # print(areasPerInitialIndex)
        return False,areasPerInitialIndex
    else:
        # print('Symmetric grow')
        # print(areasPerInitialIndex)
        return True,areasPerInitialIndex

def wulffLike(atoms,idealWulffAreasFraction,areasPerInitialIndex):
    """Function that calculates the wulff-like index,
    defined as the absolute value of the diference of 
    areas normalized by equivalent planes.
    Args:
        idealWulffAreasFraction([index,areas]):list of areas fraction present and areas per each index
        areasPerInitialIndex:([index,area]): list of present and areas per each index
    return: 
        order(bool): true if the planes have the same area contribution in ideal and real np, false if not 
        wli(float) wulff-like index. Close to 0 means that NP are close to wulffShape
    """

    sg=Spacegroup((int(str(atoms.info['spacegroup'])[0:3])))

    wli=[]
    idealAreasPerEquivalent=[]
    realAreasPerEquivalent=[]
    ##Calculate the area per equivalent faces and sort
    for n,i in enumerate(idealWulffAreasFraction):

        # number of equivalent faces
        numberOfEquivalentFaces=len(sg.equivalent_reflections(i[0]))
        # print('equivalentFaces',numberOfEquivalentFaces)

        #Ideal
        indexStringIdeal=''.join(map(str,i[0])) 
        idealAreaPerEquivalent=i[1]/numberOfEquivalentFaces
        idealAreasPerEquivalent.append([indexStringIdeal,idealAreaPerEquivalent])

        #Real
        indexStringReal=''.join(map(str,tuple(areasPerInitialIndex[n][0].tolist())))
        realAreaPerEquivalent=areasPerInitialIndex[n][1]/numberOfEquivalentFaces
        realAreasPerEquivalent.append([indexStringReal,realAreaPerEquivalent])

    #Sorting
    idealAreasPerEquivalentSort=sorted(idealAreasPerEquivalent,key=lambda x:x[1],reverse=True)
    realAreasPerEquivalentSort=sorted(realAreasPerEquivalent,key=lambda x:x[1],reverse=True)

    # print('idealAreasPerEquivalentSort',idealAreasPerEquivalentSort)
    # print('realAreasPerEquivalentSort',realAreasPerEquivalentSort)

    for n,indexArea in enumerate(idealAreasPerEquivalent):
        if indexArea[0]==realAreasPerEquivalentSort[n][0]:
            sameOrder=True
        else:
            # print('notEqual')
            sameOrder=False
        break

    #Calculate the index
    wlindex=0
    for n,indexArea in enumerate(idealAreasPerEquivalent):
        wlindex+=abs((indexArea[1]-realAreasPerEquivalentSort[n][1])/numberOfEquivalentFaces)

    return sameOrder,"%.4f"%wlindex

def idealWulffFractions(atoms,surfaces,energies):
    """Function that calculate the ideal wulff Areas Fraction
    using the pymatgen WulffShape.
    Args:
        atoms(atoms):atoms object
        surfaces([(index)]): list of index tuples
        energy([energy]): list of energy floats
    return:
        idealWulffAreasFraction([index,areas]):list of areas fraction present and areas per each index
    """
    lattice=atoms.get_cell()

    tupleMillerIndexes=[]
    for index in surfaces:
        tupleMillerIndexes.append(tuple(index))

    idealWulffShape = WulffShape(lattice,tupleMillerIndexes, energies)
    areas=idealWulffShape.area_fraction_dict

    idealWulffAreasFraction=[]
    for millerIndex,areaFraction in areas.items():
        idealWulffAreasFraction.append([millerIndex,areaFraction])
    # print(idealWulffAreasFraction)
    return idealWulffAreasFraction 
def coulombEnergy(symbol,atoms):
    """
    Function that calculate the coulomb like energy
    E=sum((qiqj/d(i,j)))for np.
    Args:
        symbol(Atoms): Crystal atoms object
        atoms(Atoms): Atoms object
    Return:
        coulombLikeEnergy(float): coulomb like energy
    """
    #Add the charges
    for iatom in atoms:
        for jatom in symbol:
            if iatom.symbol==jatom.symbol:
                iatom.charge=jatom.charge

    #Calculating the energy
    coulombLikeEnergy=0
    for c in combinations(atoms,2):
        #coulomb
        tempCoulomb=(c[0].charge*c[1].charge)/euclidean(c[0].position,c[1].position)
        # print('tempCoulomb',tempCoulomb)
        coulombLikeEnergy+=tempCoulomb

    return coulombLikeEnergy

def dipole(atoms):
    """
    Function that calculate the dipole moment
    E=sum(qi/ri))for np.
    Args:
        atoms(Atoms): Atoms object
    Return:
        dipole(float): dipole
    """
    dipole=0
    for atom in atoms:
        dipoleTemp=np.sum(atom.charge*atom.position)
        dipole+=dipoleTemp

    return dipole





