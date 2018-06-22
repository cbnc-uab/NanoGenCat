from __future__ import print_function
import numpy as np
from ase.utils import basestring
#from itertools import product
from re import findall
from ase.cluster.factory import GCD
#from ase.visualize import view
from ase.io import write
from ase.data import chemical_symbols
from random import shuffle
import copy
import os, time
from ase.neighborlist import NeighborList
#from ase import Atom, Atoms
from ase.atoms import symbols2numbers

nonMetals = ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
                  'Si', 'P', 'S', 'Cl', 'Ar',
                  'Ge', 'As', 'Se', 'Br', 'Kr',
                  'Sb', 'Te', 'I', 'Xe',
                  'Po', 'At', 'Rn']
nonMetalsNumbers=symbols2numbers(nonMetals)

delta = 1e-10
_debug = False


def wulff_construction(symbol, surfaces, energies, size, structure,
                       rounding='closest', latticeconstant=None, 
                       debug=False, maxiter=100,center=[0.,0.,0.],option=0):
    """Create a cluster using the Wulff construction.

    A cluster is created with approximately the number of atoms
    specified, following the Wulff construction, i.e. minimizing the
    surface energy of the cluster.

    Parameters:
    -----------

    symbol: The chemical symbol (or atomic number) of the desired element.

    surfaces: A list of surfaces. Each surface is an (h, k, l) tuple or
    list of integers.

    energies: A list of surface energies for the surfaces.

    size: The desired number of atoms.

    structure: The desired crystal structure.  Either one of the strings
    "fcc", "bcc", "sc", "hcp", "graphite"; or one of the cluster factory
    objects from the ase.cluster.XXX modules.

    rounding (optional): Specifies what should be done if no Wulff
    construction corresponds to exactly the requested number of atoms.
    Should be a string, either "above", "below" or "closest" (the
    default), meaning that the nearest cluster above or below - or the
    closest one - is created instead.

    latticeconstant (optional): The lattice constant.  If not given,
    extracted from ase.data.

    debug (optional): If non-zero, information about the iteration towards
    the right cluster size is printed.
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
            from ase.cluster.hexagonal import \
                HexagonalClosedPacked as structure
        elif structure == 'graphite':
            from ase.cluster.hexagonal import Graphite as structure
        elif structure == 'ext':
            from ase.cluster.cut_cluster import CutCluster as structure
        else:
            error = 'Crystal structure %s is not supported.' % structure
            raise NotImplementedError(error)

    # Check number of surfaces
    nsurf = len(surfaces)
    if len(energies) != nsurf:
        raise ValueError('The energies array should contain %d values.'
                         % (nsurf,))

    # We should check that for each direction, the surface energy plus
    # the energy in the opposite direction is positive.  But this is
    # very difficult in the general case!

    # Before starting, make a fake cluster just to extract the
    # interlayer distances in the relevant directions, and use these
    # to "renormalize" the surface energies such that they can be used
    # to convert to number of layers instead of to distances.
    ##THIS IS A 5X5X5 CLUSTER ONLY TO GET THE INTERLAYER DISTANCE
    distances = None
    atoms = structure(symbol, surfaces, 5 * np.ones(len(surfaces), int), distances,
                      latticeconstant=latticeconstant)
    
    for i, s in enumerate(surfaces):
        ##FROM BASE
        d = atoms.get_layer_distance(s,12)/12
        ##ENERGY IS NORMALISES WRT THE INTERLAYER DISTANCE SO THE
        ##PROPORTIONALITY IS E-LAYERS (UNITS OF E/N_layers)
        ##print("s",s,"get_layer_distance",d)
        #energies[i] /= d
    
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
            small = np.array(energies)/((max(energies)*2.))
            large = np.array(energies)/((min(energies)*2.))
            midpoint = (large+small)/2.
            distances = midpoint*size
            layers= distances/d
            atoms_midpoint = make_atoms_dist(symbol, surfaces, layers, distances, 
                                        structure, center, latticeconstant)
        iteration = 0
        maxiteration = 20
        while abs(np.mean(atoms_midpoint.get_cell_lengths_and_angles()[0:3]) - size) > 0.15*size:
            midpoint = (small+large)/2.
            distances = midpoint*size
            layers= distances/d
            atoms_midpoint = make_atoms_dist(symbol, surfaces, layers, distances, 
                                             structure, center, latticeconstant)
            print("ATOMS_MIDPOINT",atoms_midpoint)
            if np.mean(atoms_midpoint.get_cell_lengths_and_angles()[0:3]) > size:
                large = midpoint
            elif np.mean(atoms_midpoint.get_cell_lengths_and_angles()[0:3]) < size:
                small = midpoint
            iteration += 1
            if iteration == maxiteration:
                print("Max iteration reached, CHECK the NP0")
                print("ATOMS_MIDPOINT",atoms_midpoint)
                break
        
        print("Initial NP",atoms_midpoint.get_chemical_formula())
    
        """
        For now I will keep it here too
        """
        name = atoms_midpoint.get_chemical_formula()+"_NP0.xyz"
        write(name,atoms_midpoint,format='xyz',columns=['symbols', 'positions'])
        """
        testing it the np0 contains metal atoms with lower coordination than the half of the maximum coordination
        """
        if check_min_coord(atoms)==True:
            print('The initial NP contain metals with coordination lower than the half of the maximum coordination')
            return None
            # raise SystemExit(0)

        if option == 0:
            if all(np.sort(symbol.get_all_distances())[:,1]-max(np.sort(symbol.get_all_distances())[:,1]) < 0.2):
                n_neighbour = max(np.sort(symbol.get_all_distances())[:,1])
            else:
                n_neighbour = None
            coordination(atoms_midpoint,debug,size,n_neighbour)
            os.chdir('../')
            return atoms_midpoint
        elif option == 1:
            print("Good Luck Danilo!")
    else:
        print("Please give the NP size as an int")

def make_atoms_dist(symbol, surfaces, layers, distances, structure, center, latticeconstant):
    #print("1distances",distances)
    layers = np.round(layers).astype(int)
    #print("1layers",layers)
    atoms = structure(symbol, surfaces, layers, distances, center= center,                   
                      latticeconstant=latticeconstant)

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
    	# # 	if 
        
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

        D = []
        print('atoms pre pop\n',atoms)
        # for i,j in enumerate(C):
        #     if j < E[i]:
        #         D.append(i)
        # for i,j in enumerate(D):
        #     atoms.pop(j-i)
        #     nearest_neighbour.pop(j-i)
        #     C = np.delete(C,j-i)
        print('atoms post pop\n',atoms)
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
            for i,j in enumerate(atoms.excess):
                if j > 0:
                    C = make_C(atoms,nearest_neighbour)
                    S = make_F(atoms,C,nearest_neighbour,debug)
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
                        tot = 0
                        for k in sorted(tmp_l):
                            atoms1.pop(k-tot)
                            nearest_neighbour1.pop(k-tot)
                            
                            C1 = np.delete(C1,k-tot)
                            E1 = np.delete(E1,k-tot)
                            tot += 1
                        atoms_only_oxygen = copy.deepcopy(atoms1)
                        del atoms_only_oxygen[[atom.index for atom in atoms1 if atom.symbol in nonMetals]]
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
                            comment = ("DC = "+str(dev*100)+' Lattice="' +
                                       ' '.join([str(x) for x in np.reshape(atoms.cell.T,
                                                        9, order='F')]) +
                                       '"')
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
            del atoms_only_oxygen[[atom.index for atom in atoms if atom.symbol  in nonMetals]]
            # del atoms_only_oxygen[[atom.index for atom in atoms if atom.symbol!='O']]
            center_of_oxygen = atoms_only_oxygen.get_center_of_mass()
            
            dev_s = np.std(abs(center_of_metal-center_of_oxygen))
            dev_p = float("{:.7f}".format(round(float(dev_s*100),7)))

        else:
            if debug == 1:
                print(len(S)," different combinations were tried resulting in", 
                      len([name for name in os.listdir('.') if os.path.isfile(name)])-1,"final NPs")
        dev_p = float("{:.7f}".format(round(float(dev_s*100),7)))
        name = atoms.get_chemical_formula()+'_NPf_'+str(dev_p)+".xyz"
        comment = ("DC = "+ str(dev_s*100) +
               str(np.reshape(coord_final,(1,4))) +
               ' Lattice="' +
               'a= '+str(atoms.cell[0,0])+
               ' b= '+str(atoms.cell[1,1])+
               ' c= '+str(atoms.cell[2,2]) +
               '"')
        
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

    return C
    

def make_F(atoms,C,nearest_neighbour,debug):
    #time_0_make_F = time.time()
    """
    Create the F vector, which contains the index of the atom to which the 
    singly coordinated oxygen atoms are linked to.
    The other values are set =len(C) for the other oxygen atoms and
    =len(C)+1 for the other atom types
    """
    if debug == 1:
        print("Start calculating combinations")
        time_F0 = time.time()
    F=[]
    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)
    for i in range(len(C)):
            if atoms.get_atomic_numbers()[i] in nonMetalsNumbers:
            # if atoms.get_atomic_numbers()[i] == (8 or 16):
                if C[i] == 1:
                    indices, offsets = nl.get_neighbors(i)
                    if len(indices) == 1:
                        F.append(indices[0])
                    else:
                        print("There is a problema....",indices)
                        exit()
                else:
                    F.append(len(C))
            else:
                F.append(len(C)+1)

    """
    A nex coordination list is created by adding the values 10 and 11
    for the higher coordination oxygen atoms and other atom types, respectively.
    This is to avoid having an error in the following loop, where the coordination
    is checked and the values len(C) and len(C)+1 would be out of bound
    """
    c = list(C)
    c.append(11)
    c.append(12)

    K=[]
    n_tests = 1000
    for i in range(n_tests):
        K.append([])
    """
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
            g = []
            r = list(range(len(F)))
            shuffle(r)
            for i in r:
                if F[i] != None:
                    if c[F[i]] == j:
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
        K[y].extend(a)

    if debug == 1:
        time_F1 = time.time()
        print("Total time to calculate combinations", round(time_F1-time_F0,5)," s")
    return K
def check_min_coord(atoms):
    """
    function that identify if the nanoparticle contain lower coordinated metals
    """
    nearest_neighbour= []
    indexes=[]

    for i in range(len(atoms.get_atomic_numbers())):
        nearest_neighbour.append(np.min([x for x in atoms.get_all_distances()[i] if x>0]))

    # print (nearest_neighbour)

    C=make_C(atoms,nearest_neighbour)
    atomIndex=[atom.index for atom in atoms]
    for i in atomIndex:
        indexes.append([i,C[i]])
    """ 
    # Get the metals and non metals
    # """
    # # nonMetalAtom=[atom.index for atom in atoms if atom.symbol in nonMetals]
    metalAtom=[atom.index for atom in atoms if atom.symbol not in nonMetals]

    # """
    # Get the coordinations for metals and non metals
    # """
    # # nonMetalsCoordinations=[i[1] for i in coordinations if i[0] in nonMetalAtom]
    metalsCoordinations=[i[1] for i in indexes if i[0] in metalAtom]

    maxCoord=np.amax(metalsCoordinations)

    for i in metalsCoordinations:
        if i < maxCoord/2:
            return True
        else:
            return False 


