from __future__ import print_function
import numpy as np
from ase.utils import basestring
from itertools import product
from re import findall
from ase.cluster.factory import GCD, GCD1
from ase.visualize import view
from ase.io import write
from ase.data import chemical_symbols, covalent_radii
from random import shuffle
import copy
import os, sys, time
from ase.neighborlist import NeighborList
from ase import Atom, Atoms

delta = 1e-10
_debug = False


def wulff_construction(symbol, surfaces, energies, size, structure,
                       rounding='closest', latticeconstant=None,
                       debug=False, maxiter=100):
    """Create a cluster using the Wulff construction.

    A cluster is created with approximately the number of atoms
    specified, following the Wulff construction, i.e. minimizing the
    surface energy of the cluster.

    Parameters:

    symbol: The chemical symbol (or atomic number) of the desired element.

    surfaces: A list of surfaces. Each surface is an (h, k, l) tuple or
    list of integers.

    energies: A list of surface energies for the surfaces.

    size: The desired number of atoms.

    structure: The desired crystal structure.  One of the strings
    "fcc", "bcc", or "sc".

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
        print('Wulff: Aiming for cluster with %i atoms (%s)' %
              (size, rounding))

        if rounding not in ['above', 'below', 'closest']:
            raise ValueError('Invalid rounding: %s' % rounding)

    # Interpret structure, if it is a string.
    if isinstance(structure, basestring):
        if structure == 'fcc':
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
        else:
            error = 'Crystal structure %s is not supported.' % structure
            raise NotImplementedError(error)

    # Check number of surfaces
    nsurf = len(surfaces)
    if len(energies) != nsurf:
        raise ValueError('The energies array should contain %d values.'
                         % (nsurf,))

    # Copy energies array so it is safe to modify it
    energies = np.array(energies)

    # We should check that for each direction, the surface energy plus
    # the energy in the opposite direction is positive.  But this is
    # very difficult in the general case!

    # Before starting, make a fake cluster just to extract the
    # interlayer distances in the relevant directions, and use these
    # to "renormalize" the surface energies such that they can be used
    # to convert to number of layers instead of to distances.
    atoms = structure(symbol, surfaces, 5 * np.ones(len(surfaces), int),
                      latticeconstant=latticeconstant)
    for i, s in enumerate(surfaces):
        ##FROM BASE
        d = atoms.get_layer_distance(s,12)/12
        ##ENERGY IS NORMALISES WRT THE INTERLAYER DISTANCE SO THE
        ##PROPORTIONALITY IS E-LAYERS (UNITS OF E/N_layers)
        ##print("s",s,"get_layer_distance",d)
        #energies[i] /= d
    
    if type(size) == float:
        """
        if option == 1:
            for shift_x in [0,0.5]:
                for shift_y in [0,0.5]:
                    for shift_z in [0,0.5]:
                        center = [float(shift_x),float(shift_x),float(shift_x)]
        """
        #print("SURFACES",surfaces)
        #print("size is a float")
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
            #print("midpoint1",midpoint,"distances",distances)
            layers= distances/d
            atoms_midpoint = make_atoms_dist(symbol, surfaces, layers, distances, 
                                        structure, center, latticeconstant)
            #print("ATOMS_MIDPOINT_suerte",atoms_midpoint)
            #if atoms_midpoint.get_cell_lengths_and_angles()[0] == size:
                #return atoms_midpoint
        #print("small",small,"large",large)
        #print("MIDPOINT",midpoint)
        iteration = 0
        maxiteration = 20
        while abs(np.mean(atoms_midpoint.get_cell_lengths_and_angles()[0:3]) - size) > 0.15*size:
            midpoint = (small+large)/2.
            #midpoint = midpoint*fact
            #print("L",np.round(large,2))
            #print("S",np.round(small,2))
            #print("M",np.round(midpoint,2))
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
            
        #exit()        
        
        
        print("Initial NP",atoms_midpoint.get_chemical_formula())
        #print("Is it Stoich?",atoms_midpoint.stoichiometry)
        #return atoms_midpoint
    
        midpoint_0 =  np.copy(midpoint)
        #print("VERANO",atoms_midpoint.get_chemical_formula())
        #name = atoms_midpoint.get_chemical_formula()+"_"+str(nsurf)+"s_NP0.xyz"
        """
        For now I will keep it here too
        """
        name = atoms_midpoint.get_chemical_formula()+"_NP0.xyz"
        write_np(name,atoms_midpoint)
        """
        HERE IT IS!
        """
        """
        if options== 1:
            print("IT WORKS")
        for shift_x in [0,0.5]:
            for shift_y in [0,0.5]:
                for shift_z in [0,0.5]:
                    center = [float(shift_x),float(shift_x),float(shift_x)]
        """
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
        
        """
        THIS PART OS GOING TO BE REMOVED IN VERSION 3?
        else:
            return atoms_midpoint
            print("Start iterations to achieve stoichiometry")
            #for i in range(100):   
            for i in range(5):
            #for i in range(1000):
                for j in (1,-1):
                    for centering in (True, False):
                        for c in range(10):
                        #delta = midpoint*i*j*0.001
                            delta = midpoint*i*j*0.1
                            distances = (midpoint+delta)*size
                            layers= distances/d
                        #print("layers",layers)
                            if (layers > 1).all():
                                c = c*0.1
                                center = [c, c, c]
                                atoms_midpoint = make_stoich(symbol, surfaces, layers, distances, 
                                                             structure, centering, center, latticeconstant)
                        #print("Is it Stoich?",atoms_midpoint.stoichiometry)
                                print("NP",atoms_midpoint.get_chemical_formula(),
                                      np.mean(atoms_midpoint.get_cell_lengths_and_angles()[0:3]),
                                      "Delta",i*j*0.1,"c",c,
                                      atoms_midpoint.stoichiometry)
                            #view(atoms_midpoint)
                                if atoms_midpoint.stoichiometry == True:
                                    Delta = i*j*2
                                    #name = str(atoms_midpoint.get_chemical_formula())+"_"+str(nsurf)+"s_NP1_"+str(Delta)+"%.xyz"
                                    #write_np(name,atoms_midpoint)
                                    #return atoms_midpoint
        #return atoms_midpoint    
        print("NO STOICHIOMETRIC CLUSTER FOUND") 
        
        atoms_midpoint = None
        if atoms_midpoint == None and nsurf > 1:
            print("OH NO!")
            center = [0.5,0.5,0.5]
            #maxrange = np.array(100 * np.ones(len(surfaces), int))
            maxrange = np.array(2 * np.ones(len(surfaces), int))
            print("maxrange",maxrange)
            states = np.array([i for i in product(*(range(i+1) for i in maxrange))])
            print(len(states))
            for i in states:
                for centering in (True, False):
                    for n in (-1,1):
                        midpoint = (1+n*0.1*i)*midpoint_0
                        #midpoint = (1+n*0.01*i)*midpoint_0
                        #print("DELTA",midpoint/midpoint_0)
                        delta = np.amax(n*0.1*i*100)-np.amin(n*0.1*i*100)
                        #print("DELTA",np.amax(n*0.1*i*100),np.amin(n*0.1*i*100),np.amax(n*0.1*i*100)-np.amin(n*0.1*i*100))
                        #print("DDDDD",np.sum(abs(midpoint_0/midpoint))/3.)
                        distances = midpoint*size
                        layers= distances/d
                        if (layers > 1).all():
                            atoms_midpoint = make_stoich(symbol, surfaces, layers, distances, 
                                                    structure, centering, center, latticeconstant)
                            print("NP",atoms_midpoint.get_chemical_formula(),
                                  n*0.01*i*100, np.mean(atoms_midpoint.get_cell_lengths_and_angles()[0:3]),
                                  atoms_midpoint.stoichiometry)
                            if atoms_midpoint.stoichiometry == True:
                                #name = str(atoms_midpoint.get_chemical_formula())+"_"+str(nsurf)+"s_NP2_"+str(delta)+".xyz"
                                #write_np(name,atoms_midpoint)
                                #return atoms_midpoint
                                pass
            #return None
    
    elif type(size) == int:
        _debug = debug
        print("size is an int")
        wanted_size = size ** (1.0 / 3.0)
        max_e = max(energies)
        factor = wanted_size / max_e
        ##MAKES THE FIRST CLUSTER
        atoms, layers = make_atoms(symbol, surfaces, energies, factor, structure,
                                   latticeconstant)
        ##IF THE CLUSTER IS EMPTY IT TRIES WITH ANOTHER ENERGY
        if len(atoms) == 0:
            # Probably the cluster is very flat
            if debug:
                print('First try made an empty cluster, trying again.')
            factor = 1 / energies.min()
            atoms, layers = make_atoms(symbol, surfaces, energies, factor,
                                       structure, latticeconstant)
            if len(atoms) == 0:
                raise RuntimeError('Failed to create a finite cluster.')
    
        # Second guess: scale to get closer.
        old_factor = factor
        old_layers = layers
        old_atoms = atoms
        factor *= (size / len(atoms))**(1.0 / 3.0)
        atoms, layers = make_atoms(symbol, surfaces, energies, factor,
                                   structure, latticeconstant)
        if len(atoms) == 0:
            print('Second guess gave an empty cluster, discarding it.')
            atoms = old_atoms
            factor = old_factor
            layers = old_layers
        else:
            del old_atoms
    
        # Find if the cluster is too small or too large (both means perfect!)
        below = above = None
        if len(atoms) <= size:
            below = atoms
        if len(atoms) >= size:
            above = atoms
    
        # Now iterate towards the right cluster
        iter = 0
        while (below is None or above is None):
            if len(atoms) < size:
                # Find a larger cluster
                if debug:
                    print('Making a larger cluster.')
                ##KEEPS INCREASING THE DIMENSIONS
                factor = ((layers + 0.5 + delta) / energies).min()
                atoms, new_layers = make_atoms(symbol, surfaces, energies, factor,
                                               structure, latticeconstant)
                assert (new_layers - layers).max() == 1
                assert (new_layers - layers).min() >= 0
                layers = new_layers
            else:
                # Find a smaller cluster
                if debug:
                    print('Making a smaller cluster.')
                ##KEEPS DECREASING THE DIMENSIONS
                factor = ((layers - 0.5 - delta) / energies).max()
                atoms, new_layers = make_atoms(symbol, surfaces, energies, factor,
                                               structure, latticeconstant)
                assert (new_layers - layers).max() <= 0
                assert (new_layers - layers).min() == -1
                layers = new_layers
            if len(atoms) <= size:
                below = atoms
            if len(atoms) >= size:
                above = atoms
            iter += 1
            if iter == maxiter:
                raise RuntimeError('Runaway iteration.')
        if rounding == 'below':
            if debug:
                print('Choosing smaller cluster with %i atoms' % len(below))
            return below
        elif rounding == 'above':
            if debug:
                print('Choosing larger cluster with %i atoms' % len(above))
            return above
        else:
            assert rounding == 'closest'
            if (len(above) - size) < (size - len(below)):
                atoms = above
            else:
                atoms = below
            if debug:
                print('Choosing closest cluster with %i atoms' % len(atoms))
            return atoms

        """
def make_atoms(symbol, surfaces, energies, factor, structure, center, latticeconstant):
    ##N_layers PROPORTIONAL TO THE ENERGIES*FACTOR(iteration)
    layers1 = factor * np.array(energies)
    layers = np.round(layers1).astype(int)
    ##USES THE FACTORY TO CUT THE CLUSTER
    atoms = structure(symbol, surfaces, layers, center,
                      latticeconstant=latticeconstant)
    if _debug:
        print('Created a cluster with %i atoms: %s' % (len(atoms),
                                                       str(layers)))
    return (atoms, layers)

def make_atoms_dist(symbol, surfaces, layers, distances, structure, center, latticeconstant):
    #print("1distances",distances)
    layers = np.round(layers).astype(int)
    #print("1layers",layers)
    atoms = structure(symbol, surfaces, layers, distances, center= center,                   
                      latticeconstant=latticeconstant)
    #print("BB",atoms)
    return (atoms)

def make_stoich(symbol, surfaces, layers, distances, structure, centering, center, latticeconstant):
    #print("1distances",distances)
    layers = np.round(layers).astype(int)
    #print("1layers",layers)
    atoms = structure(symbol, surfaces, layers, distances, centering, center,
                      latticeconstant=latticeconstant)
    #print("BB",atoms)
    return (atoms)

def write_np(name1,atoms_midpoint):
    #atom_positions = ('\n'.join('  '.join(str(cell) for cell in row) for row in 
                    #atoms_midpoint.get_positions()))
    #symbols = atoms_midpoint.get_chemical_symbols()
    #print(len(symbols))
    #for i in symbols:
        #print(i), atom_positions[i]
        #print("Title")
        #for j, i in enumerate(atom_position):
            #print(int(numbers[j]),"x",*i)
        #print("******************************************")
    
    #with open(name1,'a') as f:
    #f = open(name1,'w')
    #f.write("THIS IS A HEADER \n \n")    
    write(name1,atoms_midpoint, format="xyz")
    #with open(name1,'a') as f:
        #f.write("THIS IS A HEADER")
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

    
    #atoms_0 = copy.deepcopy(atoms)
    #time_0_nn = time.time()
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
        
        
       
        
        #print("nearest_neighbour \n",nearest_neighbour)
        """
        c = time.time()
        for i in range(len(atoms.get_atomic_numbers())):
            C.append((atoms.get_all_distances()[i] <= nearest_neighbour[i] *1.2).sum()-1)
            #print("i",i,"C",C[i])
            
        print("c time:",time.time()-c)  
        """
        C = make_C(atoms,nearest_neighbour)
        
        #F= make_F(atoms,C,nearest_neighbour)
        
        
        """           
        for i in range(len(atoms.get_atomic_numbers())):
            if F[i] is not None:
                print("i",i,atoms.get_atomic_numbers()[i],F[i],C[F[i]],atoms.get_positions()[i])
        
        name = atoms.get_chemical_formula()+"_NP_test.xyz"
        write_np(name,atoms)
        """
        
        
        """
        F=[]
        for i in range(len(atoms.get_atomic_numbers())):
            index = -1
            for j in range(len(atoms.get_all_distances()[i])):
                if atoms.get_all_distances()[i][j] <= nearest_neighbour[i] *1.2:
                    index += 1
                    #print("i",i,j)
            F.append(index)
            
        for i in range(len(C)):
            print(C[i]-F[i])
        #print(f-time.time())
        exit()    
        """
        
            #C.append((np.sort((atoms.get_all_distances()))[i,1:13] <= (np.sort(atoms.get_all_distances()))[i,1]+0.2).sum())
            #print(i,C[i])
        #print("sconosciuti \n",nearest_neighbour[69], np.sort(atoms.get_all_distances()[69]),C[69])
        """First version: very slow
        startTime = time.time()
        A = np.empty((0,len(atoms.get_atomic_numbers())))
        for i in range(len(atoms.get_atomic_numbers())):
            aa=np.absolute(np.around(atoms.get_all_distances()[i],3) - np.around(nearest_neighbour[i],3)) < 0.2
            A = np.append(A,[aa],axis=0)
        for i in range(len(A)):
            C.append(np.count_nonzero(A[i]))
        print("C ",C)
        print("Loop2",len(atoms.get_atomic_numbers()),time.time()-startTime)
        """
        #time_0_array = time.time()
        coord=np.empty((0,5))
        #print("coord",coord)   
        for d in set(atoms.get_atomic_numbers()):
            a=np.array([d, np.mean([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                        np.max([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                        np.min([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d]),
                        chemical_symbols[d]])
            coord = np.append(coord,[a],axis=0)
        coord = coord[coord[:,4].argsort()]
        #print("Time coord vector",time.time()-time_0_array)
        print("coord \n",coord)
        
        if check_stoich(atoms,coord) is 'stop':
            print("Exiting because the structure is oxygen defective")
            return None
        
            
            
        """
        for i in range(len(atoms.ufu)):
            if coord[i][4] == 'O':
                a = print(coord[i])
                if 
        exit()    
        """
        """create a new vector with the max coordination/2"""
        
        #name = atoms.get_chemical_formula()+"_NP0.xyz"
        #write_np(name,atoms)
        
        #for i in range(len(atoms.get_atomic_numbers())):
            #print("AO",len(atoms.get_atomic_numbers()),i,atoms.get_atomic_numbers()[i])
        
        E=[]
        for i in atoms.get_atomic_numbers():
            if int(i) == (8 or 16):
                E.append(1)
                #print(i,E)
            else:
                for n in coord:
                    #E.append(n[2]/2)
                    if i == int(n[0]):
                    #if i == 44:
                        #print("N",n)
                        E.append((int(n[2]))/2)
                        #print(i,E)
                        #print(i,n[0],"n[2]",n[2]/2)
        
        #print("E",len(E),E)
        
        """
        for i in range(len(atomtype(size)s.get_atomic_numbers())):
            print(i,atoms.get_atomic_numbers()[i],E[i])
        """    
          
    
        """
        First version of the loop to remove undercoordinated atoms
        """
        #write_np("NP0.xyz",atoms)
        D = []
        for i,j in enumerate(C):
            if j < E[i]:
                D.append(i)
        #print("D",D)
        for i,j in enumerate(D):
            atoms.pop(j-i)
            nearest_neighbour.pop(j-i)
            C = np.delete(C,j-i)
        check_stoich(atoms)
        #view(atoms)
        #name = atoms.get_chemical_formula()+"_NP1.xyz"
        #write_np(name,atoms)
        
        """
        nearest_neighbour= []
        for i in range(len(atoms.get_atomic_numbers())): 
            nearest_neighbour.append(np.min([x for x in atoms.get_all_distances()[i] if x>0]))
        #print("nearest_neighbour \n",nearest_neighbour)
        """

        
        """
        print("*******HW*******",S)
        for i in S:
            print(atoms.get_atomic_numbers()[i],C[i])
            """
            
        """
        C=[]
        for i in range(len(atoms.get_atomic_numbers())):
            C.append((atoms.get_all_distances()[i] <= nearest_neighbour[i] *1.2).sum()-1)
            #print(i,atoms.positions[i],"\n",atoms.get_all_distances()[i],"\n",(atoms.get_all_distances()[i] <= nearest_neighbour[i] *1.2))
        """
        
        
        """
        for i in range(len(atoms.get_atomic_numbers())):
            #if atoms.get_atomic_numbers()[i] != (8 or 16): 
            print(i,"C",C[i],"E",E[i],atoms.get_atomic_numbers()[i],atoms.positions[i])
        print("**********************************************************")    
        exit()
        """
        
        """Selective removal of atoms in excess"""
        
        #write_np("NP1.xyz",atoms)
        #print("sorted C", sorted(C), sorted(range(len(C)), key= lambda k:C[k]))
        """
        I = []
        for i in range(len(S)):
            if S[i] < len(C):
                I.append(i)
        print("I",I)    
        """
        #N = sorted(range(len(C)), key= lambda k:C[k])
        """
        Example of removing a list of elements from another list
        l1 = [1,2,3,4,5]
        l2 = [2,4,5]
        print(list(set(l1) - set(l2)))
        l1 = [x for x in l1 if x not in l2]
        print(l1)
        exit()
        """
        
                
        ####C = make_C(atoms,nearest_neighbour)
        ####S = make_F(atoms,C,nearest_neighbour)
        """
        The S list contains the atoms indices (referred to the list of atoms.whatever)
        in the correct order to be removed that takes into account the coordination
        of the metal atom from which they are removed.
        First I create a list called tmp_l, which contains just the right number
        of atoms to be removed in order to obtain a stoichiometric NP 
        and then I pop them in the following step.
        (I cannot remember why I did it this way and not directly all in one step,
        but there was an error.)
        """
        ####S = make_F(atoms,C,nearest_neighbour)
        
        """
        print("S",S)
        for i in S:
            print(i,atoms.get_atomic_numbers()[i])
        """
        #total = 0.
        #print("NP0",atoms.get_center_of_mass())
        #time_0_loop = time.time()
        
        
        atoms_only_metal = copy.deepcopy(atoms)
        del atoms_only_metal[[atom.index for atom in atoms if atom.symbol=='O']]
        del atoms_only_metal[[atom.index for atom in atoms if atom.symbol=='S']]
        
        center_of_metal = atoms_only_metal.get_center_of_mass()
        index = 0
        #print("I AM TESTING HERE2",center_of_metal)
        #print("I AM TESTING HERE3",atoms.get_cell_lengths_and_angles()[0:3]/2)
        S = None
        #for i in range(len(C)):
            #print(i,atoms.get_atomic_numbers()[i],C[i],E[i])
        dev_s = 100000000.0
        #index=0
        if debug == 1:
            print("Atoms to be removed",atoms.excess)
        if atoms.excess is not None:
            for i,j in enumerate(atoms.excess):
                if j > 0:
                    #print("i",i,"j",j)
                    C = make_C(atoms,nearest_neighbour)
                    S = make_F(atoms,C,nearest_neighbour,debug)
                    for h in range(len(S)):
                        atoms1 = copy.deepcopy(atoms)
                        ###TEST
                        C1 = copy.deepcopy(C)
                        E1 = copy.deepcopy(E)
                        nearest_neighbour1 = copy.deepcopy(nearest_neighbour)
                        ind=0
                        #print("**********************************************")
                        #print("HERE2",atoms1)
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
                        ###atoms1.center()
                        #dev = np.std(atoms1.get_cell_lengths_and_angles()[0:3]/atoms1.get_center_of_mass()) 
                        #atoms1.center()
                        dev = np.std(abs(center_of_metal-atoms1.get_center_of_mass()))
                        #dev = np.std(abs(center_of_metal-atoms1.get_cell_lengths_and_angles()[0:3]/2))
                        #print("METAL",center_of_metal,"MASS",atoms1.get_center_of_mass())
                        dev_p = float("{:.7f}".format(round(float(dev*100),7)))
                        """
                        if dev_p == 0.0:
                            index += 1
                            name = atoms1.get_chemical_formula()+'_NPtest'+str(index)+".xyz"
                            write(name,atoms1,format='xyz',columns=['symbols', 'positions'])
                        """
                        #print(dev_p,center_of_metal,atoms1.get_center_of_mass())
                        #print("TEST",center_of_metal,atoms1.get_center_of_mass(),dev)
                        #dist = np.linalg.norm((atoms1.get_cell_lengths_and_angles()[0:3]/2)-atoms1.get_center_of_mass())
                        #print("DEV",dev,"NORM",dist)
                        """
                        atoms2 = Atoms('FCl', positions=[(np.ndarray.tolist(atoms1.get_cell_lengths_and_angles()[0:3]/2)),
                                                         (np.ndarray.tolist(atoms1.get_center_of_mass()))],
                                                            cell=[atoms1.get_cell()[0,0],atoms1.get_cell()[1,1],
                                                                    atoms1.get_cell()[2,2]])
                        """
                        #coc = Atom('F',position=(np.ndarray.tolist(atoms1.get_cell_lengths_and_angles()[0:3]/2)))
                        #cof = Atom('Cl',position=(np.ndarray.tolist(atoms1.get_center_of_mass())))
                        #atoms1.extend(coc)
                        #atoms1.extend(cof)
                        #print("DEV",atoms1.get_cell_lengths_and_angles()[0:3],
                              #atoms1.get_center_of_mass(),dev)
                        #print("h",dev,dev_s)
                        #print("C and E",len(C1),len(E1))
                        if debug == True:
                            #index += 1
                            comment = ("DC = "+str(dev*100)+' Lattice="' +
                                       ' '.join([str(x) for x in np.reshape(atoms.cell.T,
                                                        9, order='F')]) +
                                       '"')
                            #name = atoms1.get_chemical_formula()+'_'+str(round(dev*100,7))+"_"+str(index)+".xyz"
                            name = atoms1.get_chemical_formula()+'_'+str(dev_p)+".xyz"
                            #name = atoms1.get_chemical_formula()+'_'+str(round(dev*100,7))+".xyz"
                            #name1 = atoms1.get_chemical_formula()+'_'+str(round(dev*100,7))+".vasp"
                            #time_print = time.time()
                            write(name,atoms1,format='xyz',comment=comment,columns=['symbols', 'positions'])
                            #print("Time to print", time.time()-time_print)
                            #total += time.time()-time_print
                            #write(name1,atoms1, format="vasp", sort=True, vasp5=True)
                        if dev < dev_s:
                            #print("IS IS!!!")
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
        #print("TOTAL",total)
        #print("Time for loop", time.time()-time_0_loop)
        
        #print("STD",dev)
        #name = atoms.get_chemical_formula()+str(dev)+"_NPtest.xyz"
        #write_np(name,atoms)
        #print("TEST PRINT",len(C),len(E),len(atoms.get_atomic_numbers()))
        
        """
        for i in range(len(atoms.get_atomic_numbers())):
            #if atoms.get_atomic_numbers()[i] != (8 or 16): 
            print(i,"C",C[i],"E",E[i],atoms.get_atomic_numbers()[i],atoms.positions[i])
        print("********************************************************")
        """
        #name = atoms.get_chemical_formula()+"_NPtest.xyz"
        #write_np(name,atoms)
        #view(atoms)
        #name = atoms.get_chemical_formula()+"_NP_test.xyz"
        #write_np(name,atoms)
        
        """
        nearest_neighbour= []
        for i in range(len(atoms.get_atomic_numbers())): 
            nearest_neighbour.append(np.min([x for x in atoms.get_all_distances()[i] if x>0]))
        #print("nearest_neighbour \n",nearest_neighbour)
        """
      
        
        C = make_C(atoms,nearest_neighbour)
        """
        C=[]
        for i in range(len(atoms.get_atomic_numbers())):
            C.append((atoms.get_all_distances()[i] <= nearest_neighbour[i] *1.2).sum()-1)
            #print(i,atoms.positions[i],"\n",atoms.get_all_distances()[i],"\n",(atoms.get_all_distances()[i] <= nearest_neighbour[i] *1.2))
        """
        #name = str(atoms.get_chemical_formula())+"_NPtest.xyz"
        #write_np(name,atoms)
        for i in range(len(atoms.get_atomic_numbers())):
            #print(i,"C",C[i],"E",E[i],atoms.get_atomic_numbers()[i],atoms.positions[i],final)
            if C[i] < E[i]:
                #print(i,C[i],E[i])
                final = False
                print("final",final)
                break
          
                
    """
    for i in range(len(atoms.get_atomic_numbers())):
            #if atoms.get_atomic_numbers()[i] != (8 or 16): 
        print(i,"C",C[i],"E",E[i],atoms.get_atomic_numbers()[i],atoms.positions[i],final,"\n",
              np.sort(atoms.get_all_distances()[i]),nearest_neighbour[i])
    print("*******************************************")
    """
    
    """
    for i in range(len(atoms.get_atomic_numbers())):
            #if atoms.get_atomic_numbers()[i] != (8 or 16): 
        print(i,"C",C[i],"E",E[i],atoms.get_atomic_numbers()[i],atoms.positions[i])   
    """
    
    coord_final=np.empty((0,2))
    for d in set(atoms.get_atomic_numbers()):
        a=np.array([d, np.mean([C[i] for i in range(len(atoms.get_atomic_numbers())) if atoms.get_atomic_numbers()[i] == d])])
        coord_final = np.append(coord_final,[a],axis=0)
    #print(np.reshape(coord_final,(1,4)))
    #print("coord_final \n", coord_final)
    
    check_stoich(atoms)
    if atoms.stoichiometry == True:
        
        if S == None:
            if debug == 1:
                print("It did't go into the coordination loop, check why")
            
            dev_s = np.std(abs(center_of_metal-atoms.get_center_of_mass()))
            dev_p = float("{:.7f}".format(round(float(dev_s*100),7)))
            """
            name = atoms.get_chemical_formula()+'_NPf.xyz'
            comment = ("DC = NA"+
                   str(np.reshape(coord_final,(1,4))) +
                   ' Lattice="' +
                   'a= '+str(atoms.cell[0,0])+
                   ' b= '+str(atoms.cell[1,1])+
                   ' c= '+str(atoms.cell[2,2]) +
                   '"')
            print("Final NP", atoms.get_chemical_formula(), "| DC = NA", "| coord", coord_final[:,0], coord_final[:,1])
            #it means it didn't go into the coordination loop
            """
        
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
        #name1 = atoms.get_chemical_formula()+'_NPf_'+str(round(dev_s*100,9))+".vasp"
        #write(name1,atoms, format="vasp", sort=True, vasp5=True)
        #print("Getting here")
        #os.chdir('../')
        #print("Time coord = ", time.time()-time_0_coord)
        return atoms
    
    #print("Stoichiometric2",atoms,atoms.stoichiometry,atoms.excess)
    #exit()

    """First version
    for i in range(len(C)):
        if C[i] < 4:
            print(i)
            #print("X  ",*atoms.get_positions()[i], "#", atoms.get_chemical_symbols()[i])
            test = atoms.pop(i)
            C = np.delete(C,i)
            print(len(atoms.get_atomic_numbers()),len(C))
            #print("removed atom",i,test)
        #else:
            #print(atoms.get_chemical_symbols()[i],*atoms.get_positions()[i])
    print(atoms)
    print("Is it stoichiometric?",atoms.stoichiometry)
    write_np("NP_test.xyz",atoms)
    """
def check_stoich(atoms,coord=None):
    #time_0_check_stoich = time.time()   
    #print("GCD1",GCD1([100,20]))
    stoich_unit = np.array(findall('\d+',atoms.unit_cell_formula))
    stoich_unit = stoich_unit.astype(np.int)
    ##print("Stoich_U",stoich_unit)
    divisor_unit = GCD(stoich_unit[0],stoich_unit[1])
    ##print("divisor_unit",divisor_unit)
    unit_formula_unit = stoich_unit/divisor_unit
    
    """
    for i in range(len()):
        if coord[i][4] == 'O':
            a = print(coord[i])
     """       
       
        
    """
    atoms = self.Cluster(symbols=numbers, positions=positions, cell=cell)
    """
    ##print("MYB",atoms.get_chemical_formula())
    
    stoich_cl = np.array(findall('\d+',atoms.get_chemical_formula()))
    #print("TEST",findall('\d+',atoms.get_chemical_formula()))
    stoich_cl = stoich_cl.astype(np.int)
    #print("Cluster composition",stoich_cl,coord[0][4],coord[1][4])
    a = 0
    b = 0
    if coord is not None:
        for i in range(len(stoich_cl)):
            #print("i",i)
            if coord[i][4] == 'S' or coord[i][4] == 'O':
                a = stoich_cl[i]/unit_formula_unit[i]
                #print("a",a)
            else:
                b = stoich_cl[i]/unit_formula_unit[i]
                #print("b",b)
        if (b == 0) or (a == 0):
            return 'stop'
        if b > a:
            return 'stop'
    #print("Stoich",(stoich_cl/np.amin(stoich_cl))/unit_formula_unit)
    #print("IDEAL",ideal_cl,"DELTA",stoich_cl-ideal_cl)
    #print("MIN where?", np.argmin((stoich_cl/np.amin(stoich_cl))/unit_formula_unit))
    atoms.excess = None
    if stoich_cl.size != unit_formula_unit.size:
        #print("NO!")
        atoms.stoichiometry = False
        #self.Cluster.excess = 
    else:
        divisor_cl = GCD(stoich_cl[0],stoich_cl[1])
        unit_formula_cluster = stoich_cl/divisor_cl
        ##print("unit_formula_cluster",unit_formula_cluster)
        ##print("DIFF",-unit_formula_unit+unit_formula_cluster)
        ##print("ID",stoich_cl[0]/unit_formula_unit[0])
        if sum(abs((unit_formula_cluster - unit_formula_unit))) == 0:
            atoms.stoichiometry = True
        else:
            atoms.stoichiometry = False
            #a = min(stoich_cl)
            #print(stoich_cl)
            t= np.argmin((stoich_cl/np.amin(stoich_cl))/unit_formula_unit)
            ideal_cl = stoich_cl[t]//unit_formula_unit[t] * unit_formula_unit
            #print("unit_formula_unit",unit_formula_unit,ideal_cl)
            atoms.excess = stoich_cl - ideal_cl
    
    ##THIS IS THE FINAL CLUSTER
    #print("AA",self.Cluster(symbols=numbers, positions=positions, cell=cell))
    #print("Time check_stoich = ", time.time()-time_0_check_stoich)    
def make_C(atoms,nearest_neighbour):
    #time_0_make_C = time.time()
    C=[]
    
    #time_C0 = time.time()
    """First version of the loop
    for i in range(len(atoms.get_atomic_numbers())):
        C.append((atoms.get_all_distances()[i] <= nearest_neighbour[i] *1.2).sum()-1)
    """
    #time_C01 = time.time()
    
    """Let's test this cause I am not 100% sure"""
    half_nn = [x /2.5 for x in nearest_neighbour]
    nl = NeighborList(half_nn,self_interaction=False,bothways=True)
    nl.update(atoms)
    for i in range(len(atoms.get_atomic_numbers())):
        indices, offsets = nl.get_neighbors(i)
        C.append(len(indices))

    
    """
    C_test = []
    half_nn = []
    for i in atoms.get_atomic_numbers():
        half_nn.append(covalent_radii[i]/0.5)
    nl = NeighborList([0.7777]*len(atoms.get_atomic_numbers()),self_interaction=False,bothways=True)
    nl.update(atoms)
    for i in range(len(atoms.get_atomic_numbers())):
        indices, offsets = nl.get_neighbors(i)
        C_test.append(len(indices))
    for i in range(len(C)):
        print(C[i],C_test[i])
    
    """
    """
    for i in range(len(C)):
        print(atoms.get_atomic_numbers()[i],np.round(nearest_neighbour[i],2),covalent_radii[atoms.get_atomic_numbers()[i]]*2.5)
    print("AVERAGE",np.average(nearest_neighbour))
    """
    #print("CCC",C)
    #time_C1 = time.time()
    #print("t1",time_C01-time_C0)
    #print("t0",time.time()-time_C01)
    #print("Time to calculate coordination: ",round((time_C1-time_C0),4))
    #print("Time make_C = ", time.time()-time_0_make_C)
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
            if atoms.get_atomic_numbers()[i] == (8 or 16):
                if C[i] == 1:
                    indices, offsets = nl.get_neighbors(i)
                    if len(indices) == 1:
                        F.append(indices[0])
                        #print("III",i,indices[0])
                    else:
                        print("There is a problema....",indices)
                        exit()
                    #print(i,atoms.get_atomic_numbers()[i],C[i],nl.get_neighbors(i))
                    #for j in range(len(C)):
                        #if atoms.get_all_distances()[i][j] <= nearest_neighbour[i] *1.2 and j != i:   
                            #F.append(j)
                else:
                    #F.append(None)
                    F.append(len(C))
            else:
                #F.append(None)
                F.append(len(C)+1)
    #print("F",F)
    
    """
    B = sorted(F)
    print("B",B)
    """ 
    """
    A nex coordination list is created by adding the values 10 and 11
    for the higher coordination oxygen atoms and other atom types, respectively.
    This is to avoid having an error in the following loop, where the coordination
    is checked and the values len(C) and len(C)+1 would be out of bound
    """
    c = list(C)
    c.append(11)
    c.append(12)
    
    """
    for i in range(len(C)):
        print(i,"F",F[i],c[F[i]])
    exit()
    """
    """
    print("i,atoms.get_atomic_numbers()[i],c[i],F[i],c[F[i]]")
    for i in range(len(C)):
        print(i,atoms.get_atomic_numbers()[i],c[i],F[i],c[F[i]])
    print("***************************************************")
    """
    
    
    
    #print("Ctest",C,F)
    
    #I = []
    #for i in range(len(F)):
        #if F[i] < len(C):
        #I.append(i)
        #print(i,F[i],c[F[i]])
    
    #S = sorted(range(len(F)), key= lambda k:c[F[k]])
    
    """
    I don't know what this is
    S = []
    for i in range(len(C)):
        S.append(c[F[i]])
    """
    """
    print("************************************")
    for i in S:
        print(i,F[i],c[F[i]])
    print("************************************")
    S=[]
    """
    
    """
    S is just a list of indices going from 0 to len(C)
    """
    #S = list(range(len(C)))
    
    
    #a=[]
    #B=[]
    #A_tmp = []
    #g=[]
    K=[]
    n_tests = 1000
    #print("******************REMEMBER YOU ARE TESTING n_tests****************")
    #n_tests = 50
    for i in range(n_tests):
        K.append([])
    """
    In the first part of the loop I generate the following lists:
    g: it is a list of indices grouping together the atoms
    linked to singly coordinated atoms, which have the same coordination
    a: gives a list of indices of the order to pick from the atoms.whatever
    in order to select the atoms in g
    In the second loop the G and A lists are equivalent to the g and a lists,
    but the indices g all appear once before appearing a second time. This way
    first all O atoms linked to 6 coordinated atoms, once these lost one O
    they are 5 coordinated atoms and should go after.
    
    **************************NEED TO****************************
    2. once an atom is removed, the second time it should appear among the 
        5 (or 4) coordinated ones.
    """
    #s = list(S)
    l = [10,9,8,7,6,5,4,3,2,1,11,12]
    #l=[4,5,6,7,8,9,10,11,12]
    #print("BEGINNINGS OF THE LOOP")
    #A = []
    F_safe = copy.deepcopy(F)
    c_safe = copy.deepcopy(c)
    for y in range(n_tests):
        #print("HERE",len(B),len(C))
        a=[]
    #THERE IS A PROBLEM HERE, I AM ALWAYS REMOVING THE FIRST ATOMS FIRST
        F = copy.deepcopy(F_safe)
        c = copy.deepcopy(c_safe)
        #for j in range(1,13):
        for j in l:
            #print("**********************JJ",j)
            #ind = 0
            g = []
            r = list(range(len(F)))
            shuffle(r)
            for i in r:
                #print("I",i)
                if F[i] != None:
                    #print("Not None")
                #print(i,ind,i-ind,j,F[i-ind],c[F[i-ind]])
                    if c[F[i]] == j:
                        #print("F[i]",F[i])
                        if j < 11:
                            if F[i] not in g:
                                #print("J",j,i)
                                g.append(F[i])
                                #print("gg",g)
                                #a.append(S[i])
                                a.append(i)
                                #print("aa",a)
                                F[i] = None
                                #F.pop(i-ind)
                                #S[i] = None
                                #S.pop(i)
                                #ind += 1
                            else:
                                #This reduces the c of a Me atom if one O has already been removed
                                c[F[i]] = j-1
                                #print("c",c[F[i-ind]])
                        else:
                            g.append(F[i])
                            a.append(i)
                            F[i] = None
                            #F.pop(i-ind)
                            #S[i] = None
                            #S.pop(i-ind)
                            #ind += 1
            
        #while len(g) > 0:
            #print(j,ind,g,a)
            
        #G = []
        
        #A.extend(a)
        #print("A",len(a),a)
        K[y].extend(a)
        """SHIFT
            ind_2 = 0
            
            #!Add here a if j > 11
            for n in range(len(g)):
                if j < 11:
                    A.append(a[n-ind_2])
                else:
                    for w in range(n_tests):
                        K[w].append(a[n-ind_2])
                    #K.append(a[n-ind_2])
                G.append(g[n-ind_2])
                #A_tmp.append(a[n-ind_2])
                g.pop(n-ind_2)
                a.pop(n-ind_2)
                ind_2 += 1
            B.extend(G)
            #print("aa",A)
            for y in range(n_tests):
                shuffle(A)
                K[y].extend(A)
                #print(y,"A",A)
            #print("B",len(B),B)
            #print("B",len(B),B)
            """
    """
    #Check on the randomised set
    for i in range(len(C)):
        print(i,A_tmp[i],K[i],atoms.get_atomic_numbers()[A_tmp[i]],C[A_tmp[i]],
              atoms.get_atomic_numbers()[K[i]],C[K[i]])
    exit()
    """
    """
    for i in range(len(B)):
        print(i,A[i],c[B[i]],atoms.get_atomic_numbers()[A[i]],C[A[i]])
        #print(i,A[i],B[i],c[B[i]])    
    """
    """
    print("K")#,len(K),len(K[0]),K[0])
    for i in range(len(K)):
        print(i,len(K[i]),K[i])
    """
        
    if debug == 1:
        time_F1 = time.time()
        print("Total time to calculate combinations", round(time_F1-time_F0,5)," s")
    #print("Time make_F = ", time.time()-time_0_make_F)
    return K
    
    """
    THIS IS THE FIRST VERSION, WHERE IT WOULD WRITE THE VECTORS IN A 
    6
    6
    6
    10
    20
    6
    6
    6
    WAY, MEANING THE DUPLICATED ATOMS APPEAR AFTER THE ONES THAT SHOULD'T BE REMOVED.
    
    #print("F",F)
    while len(B) < len(C):
        #ind = 0
        #print("len",len(B),len(G))
        for j in range(1,21):
            #print("J",j)
            ind = 0
            G = []
            for i in range(len(F)):
                if F[i-ind] not in G:
                    if c[F[i-ind]] == j:
                        print("coordination",c[F[i-ind]])
                        A.append(S[i-ind])
                        G.append(F[i-ind])
                        F.pop(i-ind)
                        S.pop(i-ind)
                        ind += 1
            B.extend(G)
                
    print("*******A********",len(B),len(A))
    for i in range(len(B)):
        #print(A[i],B[i],c[B[i]])
        print(B[i],c[B[i]])
    exit()
    print("B",B)
    print("F",F)
    print("A",A)
    
    return F,S
    #print("I",I)
    
    #for i in I:
        #print(i,F[i],C[F[i]])
    
    """
    
    """
    for i in range(len(C)):
        if F[A[i]] < len(C)+1:
            print(i,"C",C[i],"A",A[i],"FA",F[A[i]],"M",C[F[A[i]]])
            
    Q = sorted(range(43), key= lambda k:C[F[A[k]]])
    print("Q",Q)
    return F
    """ 
def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r
